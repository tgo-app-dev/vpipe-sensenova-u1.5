#include "u15-generate-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "interfaces/ui-delegate-intf.h"
#include "stages/model-config-source.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/weight-set.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"

#include <cstring>
#include <vector>
#include <string>
#include <utility>

using vpipe::ConfigKey;
using vpipe::ConfigType;
using vpipe::FlexData;
using vpipe::FlexDataPayload;
using vpipe::InEdge;
using vpipe::Job;
using vpipe::PortSpec;
using vpipe::ResourceClaim;
using vpipe::RuntimeContext;
using vpipe::SessionContextIntf;
using vpipe::StageCategory;
using vpipe::StageSpec;
using vpipe::TensorBeat;
using vpipe::TensorBeatPayload;
using vpipe::fmt;

namespace u15 {

namespace {

constexpr int kPromptPort = 0;
constexpr int kModelPort  = 1;
constexpr int kConfigPort = 2;
// Reference images for EDITING. Two ports rather than one because
// multi-reference editing is a real mode of this model, and rather than
// a list because ports are static while a request is not -- the same
// reasoning the in-tree generate-image uses for ref_latent0/1.
constexpr int kRefPort0 = 3;
constexpr int kRefPort1 = 4;

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "the SenseNova-U1.5 checkpoint directory (config.json, the "
          "safetensors shards, vocab.json + merges.txt). OPTIONAL: a "
          "model-select source on the model iport overrides it",
   .suggest_db = vpipe::kModelRegistryDb,
   .suggest_db_type = "sensenova-u1.5",
   // On the shared-model channel, so a model-select source offers this
   // family too. The host cannot list it -- this stage does not exist
   // when the host is compiled -- so the picker is derived from what
   // every registered consumer declares, and this is the declaration.
   .model_channel = "diffusion-model"},

  {.key = "width", .type = ConfigType::Int, .required = false,
   .doc = "output width in pixels, rounded UP to a multiple of 32 (one "
          "LLM token covers a 32x32 block). Default 1024",
   .def_int = 1024},
  {.key = "height", .type = ConfigType::Int, .required = false,
   .doc = "output height, same rule. Default 1024", .def_int = 1024},
  {.key = "steps", .type = ConfigType::Int, .required = false,
   .doc = "denoise steps. The reference's examples use 50; 20 is "
          "already a good picture and 4 is a usable preview",
   .def_int = 50},
  {.key = "seed", .type = ConfigType::Int, .required = false,
   .doc = "initial-noise RNG seed. NOTE this does NOT reproduce the "
          "reference implementation's image at the same seed -- torch's "
          "Philox generator is not reproduced here, so the seed selects "
          "a sample from this port's own stream. Use the model-config "
          "stage's `init_noise` to compare against the reference",
   .def_int = 42},

  {.key = "i8_gemm", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated mode (LOSSY): dynamic-int8 GEMMs for the "
          "backbone's big projections instead of bf16, at int8 quality. "
          "The activation and the weight are quantized on the fly with "
          "per-512-group scales and the product runs on the matrix "
          "units' int8 pipe. Ignored on a box without them. It also "
          "self-gates on ROWS -- the crossover is ~1k, so a small canvas "
          "keeps the bf16 tiles either way -- and on how much a "
          "contraction would have to be padded to reach a whole int8 "
          "group, which this model never pays: its K is 4096 or 12288, "
          "both whole 512-groups. Default false; env VPIPE_I8_GEMM "
          "overrides",
   .def_bool = false},
  // THE SAME WORDS generative-models/shared/accel-settings.h uses, and
  // deliberately: this stage owns its own config -- it is not
  // generate-video, so no acceleration bag reaches it -- but a user who
  // knows the key on one should not have to learn a second name for the
  // same tier here.
  {.key = "sage_attn", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated mode (LOSSY): SageAttention, the flash "
          "attention's QK^T product in int8 with one scale per "
          "attention block and the key side quantized as K - mean(K) "
          "over tokens. That smoothing is exact rather than "
          "approximate -- a per-channel shift moves every score in a "
          "row by the same amount and softmax does not see it -- and "
          "P*V stays in bf16. It reaches the BIDIRECTIONAL pass, which "
          "is the one that runs on the flash kernel; the causal and "
          "block-causal branches take their own kernels and are "
          "unaffected. Matrix cores only, and ignored with a message "
          "on a box without them: the int8 fragment MMA has no ALU "
          "fallback. Independent of i8_gemm and settable with it -- "
          "that one changes how a weight is multiplied, this one how a "
          "score is computed. Default false; env VPIPE_SAGE_ATTN "
          "overrides",
   .def_bool = false},
  {.key = "sage_dense_layers", .type = ConfigType::Int, .required = false,
   .doc = "leading backbone layers left in bf16 when sage_attn is on. "
          "Zero by default: Sage computes every key and every query, so "
          "unlike a method that DROPS keys there is no published reason "
          "to protect an early layer's less redundant residual stream. "
          "It exists so a caller who measures one can act on it",
   .def_int = 0},
  {.key = "unload_when_idle", .type = ConfigType::String, .required = false,
   .doc = "what to do with ~32 GB of weights between beats: 'keep' (the "
          "default -- a second prompt then costs no reload), 'park' "
          "(hand them to the kernel as purgeable: they survive unless "
          "something else needs the RAM, and the next beat takes them "
          "back without touching the disk), 'destroy' (free them; the "
          "next beat pays a full reload), or 'auto' (park when the box "
          "is tight, keep when it is not -- decided at the first beat, "
          "where every peer has loaded). NOTE park only reclaims what "
          "the weight set CACHED. MEASURED: a w8 pack parks 94% of "
          "itself, a bf16 one 55% (the generation expert is a dtype "
          "conversion, and derived entries are not parkable), and a "
          "STREAMING model only its trunk and pinned prefix. The log "
          "says how much every time, including when it is nothing",
   .def_str = "keep"},
};

const PortSpec kIports[] = {
  {.name = "prompt",
   .doc = "the prompt (FlexData string, or an object with `text`)",
   .type = &typeid(FlexDataPayload), .tags = "text", .clock_group = 0},
  {.name = "model",
   .doc = "OPTIONAL model reference (model-select); overrides hf_dir",
   .type = &typeid(FlexDataPayload), .tags = "model", .clock_group = 0},
  {.name = "model_config",
   .doc = "OPTIONAL sampling parameters from sensenova-u1.5-model-config. "
          "A beat naming a DIFFERENT family is reported and ignored "
          "rather than half-applied",
   .type = &typeid(FlexDataPayload), .tags = "model-config",
   .clock_group = 0},
  {.name = "reference0",
   .doc = "OPTIONAL reference image for EDITING, as a planar U8 RGB "
          "TensorBeat [3,H,W] -- wire load-image straight to it. Any "
          "size: the stage does the model's own smart_resize to a "
          "multiple of 32 and its ImageNet normalisation, which are "
          "model facts and not a graph's business. Wiring this switches "
          "the run from text-to-image to editing",
   .type = &typeid(TensorBeatPayload), .tags = "rgb-frames",
   .clock_group = 0},
  {.name = "reference1",
   .doc = "OPTIONAL second reference image, for multi-reference editing "
          "(\"combine these\", \"put the subject of one into the other\"). "
          "The prompt may place them with `<image>` markers; without "
          "markers the model layer prepends them the way the reference "
          "does, NAMING them Image-1 / Image-2 when there is more than "
          "one",
   .type = &typeid(TensorBeatPayload), .tags = "rgb-frames",
   .clock_group = 0},
};

const PortSpec kOports[] = {
  {.name = "image",
   .doc = "the generated image as a planar U8 RGB TensorBeat [3,H,W] -- "
          "byte for byte the payload `vae-decode` emits, so the stock "
          "save-image and compare-image consume it unchanged",
   .type = &typeid(TensorBeatPayload), .tags = "rgb-frames",
   .clock_group = 0},
};

const StageSpec kSpec = {
  .type_name = U15GenerateStage::kTypeName,
  .doc = "SenseNova-U1.5 text-to-image: a prompt becomes an image, in "
         "one stage. The model is a Qwen3-shaped Mixture-of-Transformers "
         "with a pixel-space flow-matching head -- there is no VAE and "
         "no separate text encoder, so the conditioning is a KV cache "
         "the denoise attends to rather than a tensor that crosses a "
         "port. Emits pixels directly; wire it straight to save-image.",
  .display_name = "SenseNova-U1.5 Generate",
  .category = StageCategory::Generative,
  .iports = kIports,
  .oports = kOports,
  .attrs = kAttrs,
};

std::string
prompt_of_(const FlexData& fd)
{
  if (fd.is_object()) {
    const auto o = fd.as_object();
    if (o.contains("text")) { return std::string(o.at("text").as_string("")); }
    if (o.contains("prompt")) {
      return std::string(o.at("prompt").as_string(""));
    }
    return {};
  }
  return std::string(fd.as_string(""));
}

}  // namespace

U15GenerateStage::U15GenerateStage(const SessionContextIntf* s,
                                   std::string               id,
                                   std::vector<InEdge>       iports,
                                   FlexData                  config)
  : vpipe::TypedStage<U15GenerateStage>(s, std::move(id), std::move(iports),
                                        std::move(config))
{
  // The spec is DOCUMENTATION. A stage that skips this reports ZERO
  // oports to pipeline_from_spec, so every downstream edge is refused
  // as out of range.
  this->allocate_oports(std::size(kOports));

  // Deferred validation: the constructor never throws, and a bad config
  // is reported at launch, where the runtime skips the stage.
  _hf_dir = this->attr_str("hf_dir");
  _params.width = (int)this->attr_int("width");
  _params.height = (int)this->attr_int("height");
  _params.steps = (int)this->attr_int("steps");
  _params.seed = (std::uint64_t)this->attr_int("seed");
  _i8_gemm = this->attr_bool("i8_gemm");
  _sage_attn = this->attr_bool("sage_attn");
  _sage_dense_layers = (int)this->attr_int("sage_dense_layers");
  bool bad_policy = false;
  _policy = vpipe::model_memory::parse_unload_policy(
      this->attr_str("unload_when_idle"), &bad_policy);
  _idle = _policy;

  if (_params.steps <= 0) {
    this->fail_config(fmt("steps must be positive"));
  }
  if (_params.width <= 0 || _params.height <= 0) {
    this->fail_config(fmt("width and height must be positive"));
  }
  if (bad_policy) {
    this->fail_config(fmt(
        "unload_when_idle must be keep / park / destroy / auto"));
  }
  // NOTE: no check that _hf_dir exists. Whether the model port is wired
  // is a RUNTIME fact -- ctx.iport_connected() -- and a constructor
  // cannot see the graph, so refusing an empty hf_dir here would reject
  // a graph that legitimately gets its directory from a port.
}

const StageSpec&
U15GenerateStage::spec() const noexcept
{
  return kSpec;
}

const StageSpec*
U15GenerateStage::stage_spec() noexcept
{
  return &kSpec;
}

namespace {

// The stem every transformer layer's tensors sit under. One string,
// used by the floor, the pin count and the streaming reader, so they
// cannot disagree about what a "layer" is.
constexpr const char* kLayerStem = "language_model.model.layers.";

// A plan-time estimate of what one beat ALLOCATES, from the config
// alone -- no model load, so it is answerable during the planning
// phase.
//
// Two terms, both large and neither a weight:
//
//   the KV caches   [kv_heads][prefix + image][head_dim] per layer, per
//                   CFG branch. This is the one that grows DURING a run
//                   and that a weights-only accounting cannot see.
//   the arena       the backbone's shared activation scratch, eleven
//                   buffers at the widest pass.
//
// The prefix allowance is generous on purpose: the real one is known
// only after tokenising, and a declaration must not under-report.
struct BeatBytes {
  std::size_t kv = 0;
  std::size_t arena = 0;
};

BeatBytes
beat_bytes_(const U15Config& cfg, int width, int height, int branches)
{
  BeatBytes b;
  const int th = align_dimension(height) / kPixelsPerToken;
  const int tw = align_dimension(width) / kPixelsPerToken;
  const int img = th * tw;
  const int cap = 1024 + img;
  const std::size_t elt = 2;   // bf16
  b.kv = (std::size_t)cfg.llm.num_key_value_heads * (std::size_t)cap *
         (std::size_t)cfg.llm.head_dim * elt * 2 *
         (std::size_t)cfg.llm.num_hidden_layers * (std::size_t)branches;
  const std::size_t H  = (std::size_t)cfg.llm.hidden_size;
  const std::size_t QD = (std::size_t)cfg.llm.num_attention_heads *
                         (std::size_t)cfg.llm.head_dim;
  const std::size_t KD = (std::size_t)cfg.llm.num_key_value_heads *
                         (std::size_t)cfg.llm.head_dim;
  const std::size_t I  = (std::size_t)cfg.llm.intermediate_size;
  b.arena = (std::size_t)cap * elt * (2 * H + 3 * QD + 2 * KD + 3 * I);
  return b;
}

constexpr const char* kScratchLabel = "u15-denoise";
constexpr const char* kKvLabel      = "u15-kv";

}  // namespace

void
U15GenerateStage::apply_constant(unsigned iport, const vpipe::FlexData& beat)
{
  if (iport != (unsigned)kModelPort) { return; }
  // The REFERENCE, exactly as the runtime latch stores it -- the two
  // have to agree or the directory declared is not the one loaded.
  vpipe::apply_model_select_beat(beat, _hf_dir);
}

std::string
U15GenerateStage::model_dir_() const
{
  return vpipe::resolve_model_dir(session(), _hf_dir);
}

std::vector<ResourceClaim>
U15GenerateStage::declare_resources() const
{
  // `_hf_dir` is whichever source named the checkpoint: this stage's
  // own config, or a model-select beat that apply_constant() latched
  // before this phase runs. Empty means neither did, and a claim for a
  // model that was never named would size peers against one that never
  // appears.
  if (_hf_dir.empty()) { return {}; }

  namespace mm = vpipe::model_memory;
  std::vector<ResourceClaim> out;

  // The RESOLVED directory, not the reference: these claims are keyed
  // by directory and have to name the same one open_weight_set() will,
  // or the ledger describes a checkpoint nothing ever loads. An
  // unresolved key also measures ZERO here -- streaming_floor_bytes()
  // and weight_claims() both walk the path -- and a claim that reads as
  // small rather than absent is what admits a graph the box cannot
  // hold.
  const std::string dir = model_dir_();

  // WEIGHTS, with the floor this model can be reduced to if it streams:
  // everything outside the layer stack, plus the two in-flight slots a
  // streamer refills into. Declaring only the checkpoint's size would
  // tell a peer sizing after us that a 16 GB box cannot run this graph,
  // when in fact it can -- slowly.
  const std::size_t floor =
      mm::streaming_floor_bytes(dir, {kLayerStem});
  if (floor > 0) {
    out.push_back(mm::weight_claim_streamable(dir, floor));
  } else {
    for (auto& c : mm::weight_claims({dir})) {
      out.push_back(std::move(c));
    }
  }

  // THE TWO ALLOCATIONS THAT ARE NOT WEIGHTS. The KV cache is the one
  // that matters: it is the only large allocation that GROWS during a
  // run, so a weights-only accounting reads as healthy right up until a
  // long prompt exhausts the box.
  //
  // Reading the config here is a JSON parse of a file already on disk,
  // which is what the planning phase is for. When it cannot be read the
  // claims are simply not made -- an estimate from guessed dimensions
  // would be worse than the absence of one.
  U15Config cfg;
  std::string why;
  if (parse_config(dir, &cfg, &why)) {
    // Three branches is the EDIT worst case (conditional, image-only,
    // unconditional); a t2i run uses two. Over-declaring the KV is the
    // safe direction -- it is what the box has to survive.
    const BeatBytes b =
        beat_bytes_(cfg, _params.width, _params.height, 3);
    for (auto& c : vpipe::model_memory::scratch_claims(
             kScratchLabel, b.arena, {})) {
      out.push_back(std::move(c));
    }
    for (auto& c : vpipe::model_memory::scratch_claims(kKvLabel, b.kv, {})) {
      out.push_back(std::move(c));
    }
  }
  return out;
}

bool
U15GenerateStage::ensure_loaded_()
{
  if (_gen != nullptr) { return true; }
  if (_load_failed) { return false; }

  const auto fail = [this](const std::string& m) {
    session()->warn(fmt("U15GenerateStage('{}'): {}", this->id(), m));
    _load_failed = true;
    return false;
  };

  if (_hf_dir.empty()) {
    return fail("no checkpoint: set hf_dir or wire a model-select source "
                "to the model iport");
  }
  // Resolved ONCE for the whole load: every step below walks the
  // filesystem, and re-resolving per step would hit LMDB each time for
  // an answer that cannot change mid-load.
  const std::string dir = model_dir_();
  std::string why;
  if (!detect(dir, &why)) {
    // Names the REFERENCE the user gave, and the directory it landed on
    // when those differ -- "'sensenova/Foo' is not a checkpoint: no
    // config.json" is a puzzle when the reference resolved somewhere
    // the reader cannot see.
    const std::string what =
        (dir == _hf_dir) ? ("'" + _hf_dir + "'")
                         : ("'" + _hf_dir + "' (-> " + dir + ")");
    return fail(what + " is not a SenseNova-U1.5 MoT checkpoint: " + why);
  }
  if (!parse_config(dir, &_cfg, &why)) {
    return fail("cannot read the config: " + why);
  }

  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) { return fail("no metal-compute service"); }

  _ops = std::make_unique<MetalOps>();
  std::string err;
  if (!_ops->init(mc, &err)) { return fail("Metal init: " + err); }
  // After init: the context needs the MetalCompute the ops just stored,
  // and it loads its own kernels, so asking for the mode on a host that
  // does not ship them leaves it off rather than failing the load.
  // SAGE FIRST, because a refusal here is fatal and there is no reason
  // to have built anything before finding out. `load_for_model`
  // distinguishes "asked, and this box has no matrix cores" -- which is
  // a decline, said once, and not a failure -- from "asked, and the
  // kernels would not build", which is: an image that ran bf16 under a
  // config asking for Sage would be reported as a Sage run and its
  // numbers believed.
  {
    vpipe::genai::sage::Config sc;
    sc.enabled = _sage_attn;
    sc.dense_layers = _sage_dense_layers;
    std::string serr;
    if (!_ops->set_sage(sc, &serr)) {
      return fail("sage_attn: " + serr);
    }
    if (_ops->sage_config().enabled) {
      session()->info(fmt(
          "U15GenerateStage('{}'): SageAttention ON -- the QK product in "
          "int8, {} leading layers in bf16 (LOSSY)", this->id(),
          _ops->sage_config().dense_layers));
    } else if (_sage_attn) {
      session()->warn(fmt(
          "U15GenerateStage('{}'): sage_attn was asked for and is not "
          "active here; the attention runs bf16", this->id()));
    }
  }
  const bool i8_on = _ops->enable_i8_gemm(_i8_gemm);
  if (i8_on) {
    session()->info(fmt(
        "U15GenerateStage('{}'): accelerated mode ON -- int8 GEMMs for "
        "the projections above ~1k rows (LOSSY)", this->id()));
  } else if (_i8_gemm) {
    // Asked for and not available: say so rather than running bf16
    // silently, which reads as "the flag did nothing".
    session()->warn(fmt(
        "U15GenerateStage('{}'): i8_gemm was asked for but this host "
        "ships no int8 GEMM kernels; running bf16", this->id()));
  }

  auto ws = vpipe::genai::open_weight_set(dir, session());
  if (ws == nullptr) { return fail("cannot open the weight set"); }

  // ---- stream, or hold the stack? -----------------------------------
  //
  // THE ONE DECISION HERE THAT CANNOT BE WALKED BACK. It is an argument
  // to the loader, so changing the answer afterwards means destroying
  // and rebuilding a 32 GB model -- which is why it gets the wider
  // kStreamHeadroom rather than the revisable kHeadroom. Failing to
  // stream when it should have means thrash or an OOM kill; streaming
  // needlessly costs disk reads that BlockResidency then earns back as
  // the box allows.
  //
  // No encoder directory: everything this model needs is in one
  // checkpoint, so the second argument is empty and the plan is about
  // this stage's own weights beside whatever its peers declared.
  namespace mm = vpipe::model_memory;
  const mm::StreamPlan plan =
      mm::plan_streaming(session(), dir, "", mm::kStreamHeadroom);

  U15Weights::Options opt;
  opt.stream_layers = plan.stream;
  opt.wire_resident = true;
  const BeatBytes beat =
      beat_bytes_(_cfg, _params.width, _params.height, 3);
  if (plan.stream) {
    // NO PINNED PREFIX. `plan.pin_frac` still offers one, and the
    // in-tree DiTs have all stopped taking it: a prefix sized before
    // the run from a fraction of total RAM cannot sense the machine,
    // and BlockResidency -- which grows by MEASURING and sheds when it
    // finds its own pages outside RAM -- reaches a better answer from
    // zero within a pass or two. Pinning on top of it only reserves
    // room the measurement is not allowed to give back.
    session()->info(fmt(
        "U15GenerateStage('{}'): STREAMING the layers -- {} MB of "
        "weights would have to sit beside {} MB of activations and KV "
        "on a {} MB box. Layers are read per pass and kept as free "
        "memory allows",
        this->id(), plan.footprint >> 20,
        (beat.arena + beat.kv) >> 20, mm::phys_ram() >> 20));
  }

  vpipe::UiProgress prog =
      session()->open_progress("Loading SenseNova-U1.5");
  prog.update(0, 3, "weights");
  _weights = U15Weights::load(ws, mc, _cfg, opt, &err);
  if (_weights == nullptr) { return fail("weights: " + err); }

  prog.update(1, 3, "backbone");
  _backbone = U15Backbone::create(_ops.get(), _cfg, _weights.get(), &err);
  if (_backbone == nullptr) { return fail("backbone: " + err); }
  _image = ImagePath::create(_ops.get(), _cfg, _weights.get(), &err);
  if (_image == nullptr) { return fail("image path: " + err); }

  prog.update(2, 3, "tokenizer");
  _prompt = Prompt::load(dir, session(), &err);
  if (_prompt == nullptr) { return fail("tokenizer: " + err); }

  Generator::Deps d;
  d.ops = _ops.get();
  d.weights = _weights.get();
  d.backbone = _backbone.get();
  d.image = _image.get();
  d.prompt = _prompt.get();
  _gen = Generator::create(d, _cfg, &err);
  if (_gen == nullptr) { return fail("generator: " + err); }
  prog.update(3, 3, "ready");

  // REVISE the declaration down to what is actually held.
  //
  // declare_resources() sizes from the checkpoint on DISK -- 47.9 GB --
  // because that is all it can know before loading. But the generation
  // expert ships F32 and is converted to bf16 at bind, so the resident
  // set is ~31.6 GB. Leaving the declaration at the disk figure
  // over-reserves 16 GB against every peer that sizes after this stage,
  // on a box where that is the difference between streaming and not.
  //
  // revise_declaration refuses to CREATE, so this is only ever a
  // correction to a claim declare_resources() already made -- a stage
  // whose directory arrived on a port declared nothing and gets nothing
  // here either.
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->revise_declaration(dir, _weights->resident_bytes());
    // And the two non-weight terms, now that the geometry is settled.
    // The plan declared them from the config; these are the same
    // formulas over the same numbers, so they will usually agree -- the
    // point is that the arena is revised again below, from what was
    // really allocated, once a beat has run.
    mgr->revise_scratch(kScratchLabel, beat.arena);
    mgr->revise_scratch(kKvLabel, beat.kv);
  }

  // 4-bit is MEASURED not to work on this model, and it fails as a
  // picture rather than as an error -- so the run is allowed to proceed
  // (a differently-made pack might be fine) but is told plainly.
  if (_weights->quant_bits() == 4) {
    session()->warn(fmt(
        "U15GenerateStage('{}'): this pack is 4-bit (w4g{}). MEASURED on "
        "the released checkpoint, plain affine 4-bit produces STRUCTURED "
        "GARBAGE here -- mean|diff| 61-79 of 255 against bf16 and "
        "correlation 0.47-0.54, at BOTH group 32 and group 64, while "
        "8-bit is 4.6-5.0 and visually indistinguishable. Expect a "
        "broken image unless this pack was made some other way",
        this->id(), _weights->quant_group()));
  }
  if (_weights->quant_bits() > 0 && !_ops->quant_available()) {
    return fail("the pack is quantized but this host ships no affine qmm "
                "kernels");
  }

  session()->info(fmt(
      "U15GenerateStage('{}'): loaded {} -- {:.2f} GB resident ({} of {} "
      "layers held, {} MB each), of which {:.2f} GB converted from F32",
      this->id(), dir,
      (double)_weights->resident_bytes() / (1024.0 * 1024 * 1024),
      _weights->resident_layers(), _cfg.llm.num_hidden_layers,
      _weights->layer_bytes() >> 20,
      (double)_weights->converted_bytes() / (1024.0 * 1024 * 1024)));
  return true;
}

void
U15GenerateStage::resolve_policy_()
{
  namespace mm = vpipe::model_memory;
  if (_policy != mm::UnloadPolicy::kAuto || _policy_resolved) { return; }
  _policy_resolved = true;
  // Asked HERE, after a beat, because that is the first moment every
  // peer has loaded and the manager's numbers are real bytes rather
  // than declarations. Asked in the constructor it would resolve
  // against whatever happened to have loaded first.
  //
  // park rather than destroy when the box is tight: same reclaim value
  // -- the kernel takes the pages if it needs them -- and no reload
  // when it does not. The one thing that would argue for destroy is a
  // peer needing the room RIGHT NOW, which park cannot guarantee; that
  // is a decision for a graph to state, not for this to guess.
  const bool tight =
      mm::bounded(session(), {model_dir_()}, mm::kHeadroom) ||
      mm::peer_streams(session());
  _idle = tight ? mm::UnloadPolicy::kPark : mm::UnloadPolicy::kKeep;
  session()->log_debug(fmt(
      "U15GenerateStage('{}'): unload_when_idle=auto resolved to '{}' "
      "({})", this->id(), mm::unload_policy_name(_idle),
      tight ? "the box is tight, or a peer is already streaming"
            : "the weights fit beside everything else"));
}

void
U15GenerateStage::unload_()
{
  // Order matters: the generator holds raw pointers into the others.
  _gen.reset();
  _image.reset();
  _backbone.reset();
  _weights.reset();
  _prompt.reset();
  _ops.reset();
}

Job
U15GenerateStage::process(RuntimeContext& ctx)
{
  // A wired model port is read BEFORE anything can return early:
  // leaving a beat unread stalls the producer.
  if (ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    if (const auto* mp =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      std::string ref;
      if (vpipe::apply_model_select_beat(mp->data, ref) && !ref.empty()) {
        // Latch the REFERENCE, the way the config path holds one, and
        // let model_dir_() resolve it at each use. Storing the resolved
        // directory here instead made this the only path that resolved
        // at all, which is what left a configured registry key going
        // straight to the filesystem as if it were a path.
        if (ref != _hf_dir) {
          _hf_dir = ref;
          unload_();
          _load_failed = false;
        }
      }
    }
  }

  // The config port is POLLED, not read: a graph without one must not
  // deadlock waiting for a beat that never comes.
  if (ctx.num_iports() > kConfigPort && ctx.iport_connected(kConfigPort) &&
      ctx.backlog(kConfigPort) > 0) {
    auto cb = co_await ctx.read(kConfigPort);
    if (const auto* cp =
            cb ? dynamic_cast<const FlexDataPayload*>(cb.get()) : nullptr) {
      const std::string want = vpipe::model_config::family_of(cp->data);
      if (!want.empty() && want != kFamily) {
        // Reported, not half-applied: a config meant for another family
        // shares key NAMES with this one, and silently taking the ones
        // that happen to match makes a different image.
        session()->warn(fmt(
            "U15GenerateStage('{}'): model_config names family '{}' but "
            "this stage is '{}'; IGNORING it and using the defaults",
            this->id(), want, kFamily));
      } else {
        _params = GenParams::from_flex(cp->data, _params);
      }
    }
  }

  // Reference images. POLLED, not read: a text-to-image graph leaves
  // these unwired and must not block on a beat that never comes. A
  // wired-but-empty port means the same thing -- no reference this beat.
  std::vector<u15::RefImage> refs;
  for (const int port : {kRefPort0, kRefPort1}) {
    if (ctx.num_iports() <= port || !ctx.iport_connected(port)) { continue; }
    if (ctx.backlog(port) == 0) { continue; }
    auto rb = co_await ctx.read(port);
    const auto* tp =
        rb ? dynamic_cast<const TensorBeatPayload*>(rb.get()) : nullptr;
    if (tp == nullptr || tp->dtype != TensorBeat::DType::U8 ||
        tp->shape.size() != 3 || tp->shape[0] != 3) {
      session()->warn(fmt(
          "U15GenerateStage('{}'): reference{} is not a planar U8 RGB "
          "[3,H,W] TensorBeat; ignoring it", this->id(),
          port - kRefPort0));
      continue;
    }
    const int rh = (int)tp->shape[1];
    const int rw = (int)tp->shape[2];
    u15::RefImage r;
    r.width = rw;
    r.height = rh;
    r.rgb.resize((std::size_t)rw * rh * 3);
    // PLANAR in, CHANNEL-LAST out. The model layer works channel-last
    // throughout; converting here keeps that one fact in one place.
    const auto* src = tp->as_u8();
    for (int y = 0; y < rh; ++y) {
      for (int x = 0; x < rw; ++x) {
        for (int c = 0; c < 3; ++c) {
          r.rgb[((std::size_t)y * rw + x) * 3 + c] =
              src[((std::size_t)c * rh + y) * rw + x];
        }
      }
    }
    refs.push_back(std::move(r));
  }

  auto beat = co_await ctx.read(kPromptPort);
  if (!beat) {
    // EOS. signal_done() and NOT a bare co_return: the driver loops
    // while !done(), so a stage that just returns is re-entered
    // immediately and forever -- a full core at 100% holding 32 GB of
    // weights.
    unload_();
    ctx.signal_done();
    co_return;
  }
  const auto* pp = dynamic_cast<const FlexDataPayload*>(beat.get());
  if (pp == nullptr) {
    session()->warn(fmt("U15GenerateStage('{}'): prompt beat is not "
                        "FlexData; dropping", this->id()));
    co_return;
  }
  const std::string text = prompt_of_(pp->data);
  if (text.empty()) {
    session()->warn(fmt("U15GenerateStage('{}'): empty prompt; dropping",
                        this->id()));
    co_return;
  }

  if (!ensure_loaded_()) { co_return; }

  const int W = align_dimension(_params.width);
  const int H = align_dimension(_params.height);

  vpipe::UiProgress prog = session()->open_progress(
      refs.empty() ? "SenseNova-U1.5 denoise"
                   : "SenseNova-U1.5 edit");
  std::string err;
  std::vector<std::uint8_t> pixels;
  const bool ok = _gen->generate(
      text, refs, _params, &pixels,
      [&prog](int step, int total) {
        prog.update((std::uint64_t)step, (std::uint64_t)total, "denoise");
      },
      &err);
  if (!ok) {
    session()->warn(fmt("U15GenerateStage('{}'): {}", this->id(), err));
    co_return;
  }

  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::U8;
  out->shape = {3, H, W};
  out->resize_contiguous(pixels.size());
  std::memcpy(out->as_u8(), pixels.data(), pixels.size());
  ++_emitted;
  co_await ctx.write(0, std::move(out));

  // The two runtime revisions, from what the beat ACTUALLY allocated
  // rather than what the config implied. The plan stays authoritative
  // about what exists; this only supplies magnitudes.
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->revise_scratch(kScratchLabel, _backbone->scratch_resident_bytes());
    mgr->revise_scratch(kKvLabel, _gen->kv_bytes(_params));
    // AND THE WEIGHTS AGAIN, because a streaming model does not stay at
    // the figure it reported after load. Residency admits layers as the
    // box allows, and on a roomy one it converges on the whole stack
    // within a pass or two -- MEASURED on the 64 GB M4 Pro: the dense
    // checkpoint loaded 5.69 GB with 6 of 42 layers pinned and held all
    // 42 by the end of the first beat.
    //
    // Leaving the load-time number in place would tell every peer that
    // sizes after this one there is 26 GB of room this model has since
    // taken back. Revised UP is the direction that matters; the whole
    // point of declaring is that nobody sizes against a fiction.
    mgr->revise_declaration(model_dir_(), _weights->resident_bytes());
  }
  if (_weights->streaming()) {
    session()->log_debug(fmt(
        "U15GenerateStage('{}'): streamed {} layer reads ({:.1f} GB) this "
        "beat; {} of {} layers now resident",
        this->id(), _weights->streamed_layers(),
        (double)_weights->streamed_bytes() / (1024.0 * 1024 * 1024),
        _weights->resident_layers(), _cfg.llm.num_hidden_layers));
  }

  resolve_policy_();
  switch (_idle) {
    case vpipe::model_memory::UnloadPolicy::kDestroy:
      unload_();
      break;
    case vpipe::model_memory::UnloadPolicy::kPark: {
      const std::size_t held = _weights->resident_bytes();
      // UNWIRE FIRST. Wired and parked are opposites -- mark_inactive()
      // refuses a wired buffer outright -- so the pool has to let go
      // before anything else can. It also has to happen while the model
      // is still here: only unwire_from_pool() decrements the pool's
      // counter, and buffers freed while wired leak their bytes for the
      // rest of the run.
      const std::size_t unwired = _weights->wire_down();
      // AND the layers residency grew into, which parking could not see
      // even if it were allowed to: they are this model's own buffers,
      // read uncached, so the weight set never held them. Left alone, an
      // idle streaming model sits on whatever it grew to -- on a roomy
      // box, the whole checkpoint.
      const std::size_t released = _weights->release_at_idle();

      // ASKING THE MANAGER TO PARK MEANS LETTING GO FIRST, and that is
      // worth doing in only one of the two modes.
      //
      // The manager owns the checkpoint and refuses to park one anything
      // is still borrowing: a borrower holds aliases of the very buffers
      // a park makes purgeable and reads them in a forward that never
      // asks the set for anything, so a park underneath it hands its
      // reader pages the kernel may discard. It cannot tell an idle live
      // model from one mid-forward, so it does not try. Parking around a
      // live model returns zero -- and that zero looks exactly like the
      // two reasons a park legitimately gives nothing back.
      //
      // STREAMING: keep the binding. What is held is the trunk plus the
      // layers residency grew into, and the release above already gave
      // back the second -- 26497 MB of 32322 in the worked example, 82%,
      // against the ~3583 MB of trunk a park could add. Letting go to
      // collect that would cost a full re-bind on the next beat,
      // including the F32 -> bf16 conversion of the generation expert
      // this checkpoint needs. Not worth it, so it is not done, and the
      // log says the trunk stayed.
      //
      // PRELOADED: let go. Nothing was promoted, so the release returned
      // zero and the weight set is the ONLY place bytes can come back
      // from. Here `park` and `destroy` differ in what the MANAGER does
      // with the bytes rather than in what this stage does with its
      // models: both let go, and park keeps them purgeable and
      // reactivated on the next read.
      const bool streaming = _weights->streaming();
      std::size_t parked = 0;
      auto* mgr = session()->services()->generative_model_manager();
      // The DIRECTORY, again: park_weights() looks its argument up in
      // the manager's weight-set map, which is keyed by canonical
      // directory. Handed a registry key it matches nothing and parks
      // zero -- silently, since parking nothing is also what a set with
      // nothing parkable returns.
      const std::string dir = model_dir_();
      if (!streaming) {
        unload_();
        if (mgr != nullptr) {
          parked = mgr->park_weights(dir);
          mgr->revise_declaration(dir, 0);
        }
      } else if (mgr != nullptr) {
        mgr->revise_declaration(dir, _weights->resident_bytes());
      }
      // Logged INCLUDING when everything is zero. A silent nothing is
      // how a policy gets believed to be working when it is not.
      session()->log_debug(fmt(
          "U15GenerateStage('{}'): idle -- unwired {} MB, released {} MB "
          "of streamed layers, {} of {} MB held", this->id(),
          unwired >> 20, released >> 20,
          streaming
              ? std::string("kept the trunk bound (a borrowed checkpoint "
                            "cannot be parked)")
              : fmt("let go and parked {} MB", parked >> 20)(),
          held >> 20));
      break;
    }
    default:
      break;
  }
  co_return;
}

}  // namespace u15
