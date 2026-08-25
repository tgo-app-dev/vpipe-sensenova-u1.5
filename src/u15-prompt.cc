#include "u15-prompt.h"

#include "common/flex-data.h"
#include "generative-models/tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;
using vpipe::FlexData;
using vpipe::genai::Tokenizer;

namespace u15 {

const char* const kSystemMessageForGen =
    "You are an image generation and editing assistant that accurately "
    "understands and executes user intent.\n\nYou support two modes:\n\n"
    "1. Think Mode:\nIf the task requires reasoning, you MUST start with "
    "a <think></think> block. Put all reasoning inside the block using "
    "plain text. DO NOT include any image tags. Keep it reasonable and "
    "directly useful for producing the final image.\n\n2. Non-Think "
    "Mode:\nIf no reasoning is needed, directly produce the final "
    "image.\n\nTask Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's "
    "intent.\n\nB. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification "
    "or transformation.\n"
    "- The result can be an edited image or a new image based on the "
    "reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified "
    "for the rendered text in the user's description, not the language "
    "of the prompt. If no language is specified, use the user's input "
    "language.";

namespace {

// ChatML, as the `neo1_0` template spells it. MPT separator style:
//   ret = (system_prompt + sep) if the system message is non-empty
//   then, per turn: role + message + sep, or just role for an open turn
constexpr const char* kSep      = "<|im_end|>\n";
constexpr const char* kSysOpen  = "<|im_start|>system\n";
constexpr const char* kUser     = "<|im_start|>user\n";
constexpr const char* kAssist   = "<|im_start|>assistant\n";

// Non-think mode ALWAYS emits an empty think block before the image
// token -- it is not conditional on anything.
constexpr const char* kThinkPad = "<think>\n\n</think>\n\n<img>";

bool
read_file_(const std::string& path, std::string* out)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { return false; }
  std::ostringstream ss;
  ss << f.rdbuf();
  *out = ss.str();
  return true;
}

void
json_escape_(const std::string& s, std::string* out)
{
  out->push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"':  *out += "\\\""; break;
      case '\\': *out += "\\\\"; break;
      case '\n': *out += "\\n";  break;
      case '\r': *out += "\\r";  break;
      case '\t': *out += "\\t";  break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          *out += buf;
        } else {
          out->push_back((char)c);
        }
    }
  }
  out->push_back('"');
}

}  // namespace

Prompt::~Prompt() = default;

bool
SpecialIds::complete() const
{
  return im_start >= 0 && im_end >= 0 && img >= 0 && think >= 0 &&
         think_end >= 0;
}

std::unique_ptr<Prompt>
Prompt::load(const std::string& dir, const vpipe::SessionContextIntf* s,
             std::string* err)
{
  const auto fail = [err](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return std::unique_ptr<Prompt>{};
  };

  const fs::path d(dir);
  std::string vocab, merges, added;
  if (!read_file_((d / "vocab.json").string(), &vocab)) {
    return fail("no vocab.json");
  }
  if (!read_file_((d / "merges.txt").string(), &merges)) {
    return fail("no merges.txt");
  }
  // added_tokens.json is optional in principle; without it the specials
  // below will not resolve and load fails there instead, which names the
  // real problem.
  read_file_((d / "added_tokens.json").string(), &added);

  // ---- synthesise the fast tokenizer.json ---------------------------
  //
  // vocab.json is ALREADY a {piece: id} JSON object, which is exactly
  // what model.vocab wants, so it is spliced in as raw text rather than
  // parsed and re-emitted -- 3.4 MB of string building avoided, and no
  // chance of a re-encoding changing a piece.
  std::vector<std::pair<std::string, int>> added_pairs;
  std::string json;
  json.reserve(vocab.size() + merges.size() * 2 + 4096);
  // The GPT-2 / Qwen pre-tokenizer regex, verbatim from a Qwen
  // tokenizer.json. WITHOUT it the BPE runs over whole strings and
  // whitespace merges wrongly: "  leading" becomes one double-space
  // token instead of [" ", " leading"], because the `\s+(?!\S)`
  // alternative is what detaches all but the last space of a run.
  json +=
      "{\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
      "{\"type\":\"Split\",\"pattern\":{\"Regex\":"
      "\"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?"
      "\\\\p{L}+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|"
      "\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+\"},"
      "\"behavior\":\"Isolated\",\"invert\":false},"
      "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,"
      "\"trim_offsets\":true,\"use_regex\":false}]},";
  json += "\"added_tokens\":[";
  {
    bool first = true;
    if (!added.empty()) {
      FlexData a;
      try { a = FlexData::from_json(added); } catch (...) {}
      if (a.is_object()) {
        const auto o = a.as_object();
        for (auto it = o.begin(); it != o.end(); ++it) {
          // operator* returns the Entry BY VALUE, so it is bound to a
          // local -- `it->second` would be a member of the iterator.
          const auto e = *it;
          if (!first) { json += ','; }
          first = false;
          json += "{\"id\":";
          json += std::to_string(e.second.as_int(-1));
          json += ",\"content\":";
          json_escape_(std::string(e.first), &json);
          json += ",\"special\":true}";
          added_pairs.emplace_back(std::string(e.first),
                                   (int)e.second.as_int(-1));
        }
      }
    }
  }
  json += "],\"model\":{\"type\":\"BPE\",\"vocab\":";
  json += vocab;
  json += ",\"merges\":[";
  {
    bool first = true;
    std::istringstream ms(merges);
    std::string line;
    while (std::getline(ms, line)) {
      if (!line.empty() && line.back() == '\r') { line.pop_back(); }
      // The "#version: 0.2" header is a comment, not a merge.
      if (line.empty() || line[0] == '#') { continue; }
      if (line.find(' ') == std::string::npos) { continue; }
      if (!first) { json += ','; }
      first = false;
      json_escape_(line, &json);
    }
  }
  json += "]}}";

  auto p = std::unique_ptr<Prompt>(new Prompt());
  p->_tok = Tokenizer::from_huggingface_string(json, "sensenova-u1.5", s);
  if (p->_tok == nullptr) {
    return fail("could not build a tokenizer from vocab.json + merges.txt");
  }

  // Resolve the specials by NAME rather than by hardcoded id: a fork
  // that renumbers them would otherwise tokenise to a plausible prompt
  // made of the wrong tokens.
  struct Want { const char* name; int* out; };
  const Want want[] = {
      {"<|im_start|>", &p->_ids.im_start},
      {"<|im_end|>", &p->_ids.im_end},
      {"<img>", &p->_ids.img},
      {"</img>", &p->_ids.img_end},
      {"<IMG_CONTEXT>", &p->_ids.img_ctx},
      {"<think>", &p->_ids.think},
      {"</think>", &p->_ids.think_end},
  };
  for (const Want& w : want) {
    const std::int32_t id = p->_tok->special_token_id(w.name);
    if (id >= 0) { *w.out = (int)id; }
  }
  // Longest first, so a greedy left-to-right scan cannot match a short
  // token that is a prefix of a longer one.
  std::sort(added_pairs.begin(), added_pairs.end(),
            [](const auto& a, const auto& b) {
              return a.first.size() > b.first.size();
            });
  p->_added = std::move(added_pairs);

  if (!p->_ids.complete()) {
    return fail("the checkpoint's added_tokens.json is missing one of "
                "<|im_start|> <|im_end|> <img> <think> </think>");
  }
  return p;
}

std::string
Prompt::conditional_text(const std::string& prompt) const
{
  std::string s;
  s += kSysOpen;
  s += kSystemMessageForGen;
  s += kSep;
  s += kUser;
  s += prompt;
  s += kSep;
  s += kAssist;          // an OPEN turn: role only, no separator
  s += kThinkPad;
  return s;
}

std::string
Prompt::unconditional_text() const
{
  // The system message is EMPTY here, and the MPT style emits nothing at
  // all for an empty system -- not an empty system block. That is what
  // makes this prefix 13 tokens rather than 17.
  std::string s;
  s += kUser;
  s += kSep;
  s += kAssist;
  s += kThinkPad;
  return s;
}

namespace {

// Replace the FIRST occurrence of `from` in `s`.
bool
replace_first_(std::string& s, const std::string& from,
               const std::string& to)
{
  const auto at = s.find(from);
  if (at == std::string::npos) { return false; }
  s.replace(at, from.size(), to);
  return true;
}

std::size_t
count_occurrences_(const std::string& s, const std::string& what)
{
  std::size_t n = 0, at = 0;
  while ((at = s.find(what, at)) != std::string::npos) {
    ++n;
    at += what.size();
  }
  return n;
}

constexpr const char* kImagePlaceholder = "<image>";

}  // namespace

std::string
Prompt::edit_conditional_text(const std::string& prompt,
                              const std::vector<RefImage>& imgs) const
{
  std::string p = prompt;
  const std::size_t have = count_occurrences_(p, kImagePlaceholder);
  if (imgs.size() > have) {
    const std::size_t missing = imgs.size() - have;
    if (have == 0 && imgs.size() > 1) {
      // Several images and no markers: the reference NAMES them, and
      // the names are tokens the model sees.
      std::string pre;
      for (std::size_t i = 0; i < imgs.size(); ++i) {
        pre += "Image-" + std::to_string(i + 1) + ":" + kImagePlaceholder +
               "\n";
      }
      p = pre + p;
    } else {
      std::string pre;
      for (std::size_t i = 0; i < missing; ++i) {
        pre += std::string(kImagePlaceholder) + "\n";
      }
      p = pre + p;
    }
  }

  std::string q = conditional_text(p);
  for (const RefImage& im : imgs) {
    std::string blob = "<img>";
    for (int i = 0; i < im.tokens(); ++i) { blob += "<IMG_CONTEXT>"; }
    blob += "</img>";
    replace_first_(q, kImagePlaceholder, blob);
  }
  return q;
}

std::string
Prompt::edit_image_only_text(const std::vector<RefImage>& imgs) const
{
  // The reference builds this from `'<image>' * len(images)` with the
  // DEFAULT (empty) system message -- so no system block is emitted at
  // all, exactly as for the unconditional prefix.
  std::string p;
  for (std::size_t i = 0; i < imgs.size(); ++i) { p += kImagePlaceholder; }

  std::string q;
  q += kUser;
  q += p;
  q += kSep;
  q += kAssist;
  q += kThinkPad;

  for (const RefImage& im : imgs) {
    std::string blob = "<img>";
    for (int i = 0; i < im.tokens(); ++i) { blob += "<IMG_CONTEXT>"; }
    blob += "</img>";
    replace_first_(q, kImagePlaceholder, blob);
  }
  return q;
}

void
Prompt::thw_indexes(const std::vector<int>& ids,
                    const std::vector<RefImage>& imgs, std::vector<int>* t,
                    std::vector<int>* h, std::vector<int>* w) const
{
  const std::size_t n = ids.size();
  t->assign(n, 0);
  h->assign(n, 0);
  w->assign(n, 0);

  int cum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    // +1 just after an <img>, and +1 for anything that is not an
    // <IMG_CONTEXT>. So a run of context tokens holds t still.
    const int shift =
        (i > 0 && ids[i - 1] == _ids.img) ? 1 : 0;
    const int not_ctx = (ids[i] != _ids.img_ctx) ? 1 : 0;
    cum += shift + not_ctx;
    (*t)[i] = cum - 1;
  }

  // h/w at the context positions, per image, restarting each time.
  std::size_t img = 0;
  std::size_t seen = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (ids[i] != _ids.img_ctx) { continue; }
    while (img < imgs.size() &&
           seen >= (std::size_t)imgs[img].tokens()) {
      ++img;
      seen = 0;
    }
    if (img >= imgs.size()) { break; }
    const int tw = imgs[img].grid_w / 2;
    (*h)[i] = (int)(seen / (std::size_t)tw);
    (*w)[i] = (int)(seen % (std::size_t)tw);
    ++seen;
  }
}

std::vector<int>
Prompt::encode(const std::string& text) const
{
  // vpipe's Tokenizer::encode() deliberately does NOT scan for special
  // tokens -- its contract says the chat template splices them in. So
  // that scan lives here: walk the text, emit an added token's id
  // whenever one matches, and hand only the runs between them to the
  // BPE. Without this, "<|im_start|>" BPEs as the literal characters
  // (27, 91, 318, ...) instead of 151644, and the prompt is made of
  // plausible wrong tokens.
  //
  // Longest match wins, which is why _added is sorted by descending
  // length: "<|im_start|>" must not be shadowed by a shorter prefix.
  std::vector<int> out;
  std::size_t i = 0, run_start = 0;
  const auto flush = [&](std::size_t end) {
    if (end <= run_start) { return; }
    const auto ids =
        _tok->encode(std::string_view(text).substr(run_start,
                                                   end - run_start));
    out.insert(out.end(), ids.begin(), ids.end());
  };
  while (i < text.size()) {
    bool hit = false;
    for (const auto& a : _added) {
      if (a.first.size() <= text.size() - i &&
          std::memcmp(text.data() + i, a.first.data(), a.first.size()) ==
              0) {
        flush(i);
        out.push_back(a.second);
        i += a.first.size();
        run_start = i;
        hit = true;
        break;
      }
    }
    if (!hit) { ++i; }
  }
  flush(text.size());
  return out;
}

int
Prompt::vocab_size() const
{
  return (int)_tok->vocab_size();
}

}  // namespace u15
