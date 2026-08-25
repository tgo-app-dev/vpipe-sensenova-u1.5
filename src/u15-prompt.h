#ifndef VPIPE_U15_PROMPT_H
#define VPIPE_U15_PROMPT_H

// Turning a prompt into token ids.
//
// TWO PREFIXES, DIFFERENT LENGTHS. The conditional prefix carries the
// generation system message and the user's prompt; the unconditional one
// carries an EMPTY system message (so no system block is emitted at all)
// and an empty prompt. Their lengths differ -- 261 vs 13 for a typical
// prompt -- and the image tokens' constant `t` position IS that length.
// So the two passes of a CFG step use DIFFERENT position triples, and
// reusing one for both is a silent quality bug.
//
// THE CHECKPOINT SHIPS NO tokenizer.json. It has the "slow" trio --
// vocab.json, merges.txt, added_tokens.json -- while vpipe's Tokenizer
// reads the HF fast format. Rather than write a second BPE, this
// synthesises the fast JSON from the trio and hands it to the host's
// tokenizer, which is the GPT-2 byte-level BPE Qwen uses and is already
// verified against several models in the tree.

#include <memory>
#include <utility>
#include <string>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::genai { class Tokenizer; }

namespace u15 {

// The system message the reference uses for the CONDITIONAL prefix. The
// unconditional one gets the template default, which is empty -- not
// this text. Taken verbatim from the reference's utils.py.
extern const char* const kSystemMessageForGen;

// Special token ids, checked against the checkpoint at load rather than
// hardcoded blindly -- a fork that renumbers them would otherwise
// produce a plausible prompt made of the wrong tokens.
struct SpecialIds {
  int im_start = -1;
  int im_end   = -1;
  int img      = -1;      // <img>
  int img_end  = -1;      // </img>
  int img_ctx  = -1;      // <IMG_CONTEXT>
  int think    = -1;
  int think_end = -1;
  bool complete() const;
};

class Prompt {
 public:
  // Out-of-line: the Tokenizer is only forward-declared here, and an
  // implicit destructor would need it complete at every use site.
  ~Prompt();

  // Build from a checkpoint directory. Reads vocab.json, merges.txt and
  // added_tokens.json.
  static std::unique_ptr<Prompt> load(const std::string& dir,
                                      const vpipe::SessionContextIntf* s,
                                      std::string* err);

  // The full conditional prefix: ChatML with the generation system
  // message, the user's prompt, the assistant turn, an EMPTY think block
  // and the image-start token.
  std::string conditional_text(const std::string& prompt) const;

  // The unconditional prefix: no system block, empty user turn.
  std::string unconditional_text() const;

  std::vector<int> encode(const std::string& text) const;

  // ---- the EDIT path ----------------------------------------------

  // One reference image, by its POST-patch grid (before the 2x2 merge).
  struct RefImage {
    int grid_h = 0;   // pixels_h / patch_size
    int grid_w = 0;
    // Tokens this image contributes: grid_h*grid_w * downsample_ratio^2.
    int tokens() const { return (grid_h / 2) * (grid_w / 2); }
  };

  // The conditional edit prefix: the user's prompt with each `<image>`
  // replaced by `<img>` + `<IMG_CONTEXT>` x tokens + `</img>`.
  //
  // When the prompt carries FEWER `<image>` markers than there are
  // images, the reference prepends the rest -- and does it differently
  // for one image than for several ("Image-1:", "Image-2:", ...), which
  // is a real difference in what the model is told, not formatting.
  std::string edit_conditional_text(const std::string& prompt,
                                    const std::vector<RefImage>& imgs) const;

  // The IMAGE-conditioned prefix: the images and nothing else -- no
  // system message and no prompt text. This is the branch `img_cfg_scale`
  // guides away from, so it must carry the images and omit the words.
  std::string edit_image_only_text(const std::vector<RefImage>& imgs) const;

  // The (t, h, w) triple, by the reference's get_thw_indexes rule:
  //
  //   t   advances on every token EXCEPT an <IMG_CONTEXT>, and jumps
  //       once more just after an `<img>` -- so all of one image's
  //       context tokens share a single t;
  //   h/w are 0 everywhere except at <IMG_CONTEXT>, where they are the
  //       row and column in that image's POST-MERGE grid, restarting
  //       per image.
  //
  // The generated image tokens then sit at max(t) + 1, which is NOT the
  // token count once an image is in the prefix.
  void thw_indexes(const std::vector<int>& ids,
                   const std::vector<RefImage>& imgs, std::vector<int>* t,
                   std::vector<int>* h, std::vector<int>* w) const;

  const SpecialIds& specials() const { return _ids; }
  int vocab_size() const;

 private:
  Prompt() = default;
  std::unique_ptr<vpipe::genai::Tokenizer> _tok;
  SpecialIds _ids;
  // Every added token, longest first, for the special-splitting scan in
  // encode(). See the comment there for why that scan is the caller's
  // job and not the tokenizer's.
  std::vector<std::pair<std::string, int>> _added;
};

}  // namespace u15

#endif  // VPIPE_U15_PROMPT_H
