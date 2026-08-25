// Tokenisation and the chat template, against ids taken from the
// reference's own AutoTokenizer (tools/gen_prompt_golden.py).
//
// This has to be EXACT, not close. The image tokens' constant `t`
// position is the prefix LENGTH, so one token of drift in the template
// shifts every image token's position -- a quality bug with no error
// message. And the conditional and unconditional prefixes have
// different lengths, so they cannot share a position triple.
//
// Gated on VPIPE_U15_TEST_MODEL_PATH + VPIPE_U15_GOLDEN_DIR.

#include "u15-prompt.h"

#include "common/flex-data.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using vpipe::FlexData;

namespace {

int g_fail = 0;
int g_ran  = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  ++g_ran;
  if (!ok) { ++g_fail; }
}

std::vector<int>
ids_of(const FlexData& fd)
{
  std::vector<int> v;
  if (!fd.is_array()) { return v; }
  const auto a = fd.as_array();
  for (std::size_t i = 0; i < a.size(); ++i) {
    v.push_back((int)a.at(i).as_int(-1));
  }
  return v;
}

std::string
show(const std::vector<int>& v, std::size_t n = 10)
{
  std::string s = "[";
  for (std::size_t i = 0; i < v.size() && i < n; ++i) {
    if (i != 0) { s += ", "; }
    s += std::to_string(v[i]);
  }
  if (v.size() > n) { s += ", ..."; }
  return s + "]";
}

// Report WHERE two id streams diverge. "different" is not actionable;
// "they agree for 47 tokens then the template inserted one" is.
std::string
first_diff(const std::vector<int>& got, const std::vector<int>& want)
{
  const std::size_t n = std::min(got.size(), want.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (got[i] != want[i]) {
      return "first differs at index " + std::to_string(i) + ": got " +
             std::to_string(got[i]) + ", want " + std::to_string(want[i]);
    }
  }
  if (got.size() != want.size()) {
    return "identical for " + std::to_string(n) + " tokens, then lengths "
           "differ (" + std::to_string(got.size()) + " vs " +
           std::to_string(want.size()) + ")";
  }
  return "identical";
}

}  // namespace

int
main()
{
  const char* mp = std::getenv("VPIPE_U15_TEST_MODEL_PATH");
  const char* gp = std::getenv("VPIPE_U15_GOLDEN_DIR");
  if (mp == nullptr || *mp == '\0' || gp == nullptr || *gp == '\0') {
    std::printf("SKIPPED: VPIPE_U15_TEST_MODEL_PATH / "
                "VPIPE_U15_GOLDEN_DIR unset -- this test did NOT run.\n");
    return 0;
  }

  std::string txt;
  {
    std::ifstream f(std::string(gp) + "/prompt.json", std::ios::binary);
    if (!f) {
      std::printf("SKIPPED: no prompt.json golden (run "
                  "tools/gen_prompt_golden.py). NOTHING was checked.\n");
      return 0;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    txt = ss.str();
  }
  FlexData g;
  try { g = FlexData::from_json(txt); }
  catch (...) { std::printf("bad golden json\n"); return 1; }
  const auto go = g.as_object();

  std::string err;
  auto p = u15::Prompt::load(mp, nullptr, &err);
  check(p != nullptr,
        "tokenizer built from vocab.json + merges.txt" +
            (err.empty() ? "" : " (" + err + ")"));
  if (p == nullptr) { return 1; }

  // ---- special ids -------------------------------------------------
  {
    const FlexData sf = go.at("specials");
    const auto so = sf.as_object();
    const auto want = [&](const char* k) {
      return (int)so.at(k).as_int(-1);
    };
    const auto& s = p->specials();
    check(s.im_start == want("<|im_start|>"), "<|im_start|> id");
    check(s.im_end == want("<|im_end|>"), "<|im_end|> id");
    check(s.img == want("<img>"), "<img> id");
    check(s.img_end == want("</img>"), "</img> id");
    check(s.img_ctx == want("<IMG_CONTEXT>"), "<IMG_CONTEXT> id");
    check(s.think == want("<think>"), "<think> id");
    check(s.think_end == want("</think>"), "</think> id");
  }

  // ---- bare strings, to localise a BPE mismatch ---------------------
  {
    const FlexData bf = go.at("bare");
    const auto bo = bf.as_object();
    for (auto it = bo.begin(); it != bo.end(); ++it) {
      const auto e = *it;
      const std::string s(e.first);
      const std::vector<int> want = ids_of(e.second);
      const std::vector<int> got = p->encode(s);
      check(got == want, "encode(" + s + ") " +
                             (got == want ? show(got)
                                          : first_diff(got, want)));
    }
  }

  // ---- the two prefixes --------------------------------------------
  for (const char* which : {"cond", "uncond"}) {
    const FlexData f = go.at(which);
    const auto o = f.as_object();
    const std::string want_text(o.at("text").as_string(""));
    const std::vector<int> want = ids_of(o.at("ids"));

    const std::string got_text =
        (std::string(which) == "cond")
            ? p->conditional_text("a red fox sitting in snow, "
                                  "photorealistic")
            : p->unconditional_text();
    check(got_text == want_text,
          std::string(which) + ": template text matches the reference");
    if (got_text != want_text) {
      std::printf("       got:  %s\n", got_text.substr(0, 120).c_str());
      std::printf("       want: %s\n", want_text.substr(0, 120).c_str());
    }

    const std::vector<int> got = p->encode(got_text);
    check(got == want, std::string(which) + ": " +
                           std::to_string(want.size()) + " token ids " +
                           (got == want ? "EXACT" : first_diff(got, want)));
  }

  // The lengths must DIFFER -- that difference is the image tokens'
  // position offset, and a template that made them equal would be
  // subtly wrong in a way nothing downstream reports.
  {
    const std::vector<int> c = p->encode(
        p->conditional_text("a red fox sitting in snow, photorealistic"));
    const std::vector<int> u = p->encode(p->unconditional_text());
    check(c.size() != u.size(),
          "the two prefixes have DIFFERENT lengths (" +
              std::to_string(c.size()) + " vs " + std::to_string(u.size()) +
              ") -- so the CFG pair needs two position triples");
  }

  // ---- the EDIT prefixes -------------------------------------------
  //
  // Three cases, and for each the ids AND the (t,h,w) triple, because
  // the triple is where the edit path stops resembling t2i: an image's
  // context tokens share ONE t, so the generated image's position is
  // max(t)+1 and NOT the token count.
  struct EditCase {
    const char* key;
    const char* prompt;
    bool image_only;
    std::vector<u15::Prompt::RefImage> imgs;
  };
  const std::vector<EditCase> cases = {
      {"edit_cond", "<image>\nmake the fox blue", false, {{28, 40}}},
      {"edit_imgcond", "", true, {{28, 40}}},
      {"edit_cond2", "<image>\n<image>\ncombine them", false,
       {{28, 40}, {16, 16}}},
  };
  for (const EditCase& c : cases) {
    if (!go.contains(c.key)) { continue; }
    const FlexData f = go.at(c.key);
    const auto o = f.as_object();
    const std::vector<int> want = ids_of(o.at("ids"));

    const std::string txt =
        c.image_only ? p->edit_image_only_text(c.imgs)
                     : p->edit_conditional_text(c.prompt, c.imgs);
    const std::vector<int> got = p->encode(txt);
    check(got == want,
          std::string(c.key) + ": " + std::to_string(want.size()) +
              " token ids " +
              (got == want ? "EXACT" : first_diff(got, want)));
    if (got != want) { continue; }

    std::vector<int> t, h, w;
    p->thw_indexes(got, c.imgs, &t, &h, &w);
    const std::vector<int> wt = ids_of(o.at("t"));
    const std::vector<int> wh = ids_of(o.at("h"));
    const std::vector<int> ww = ids_of(o.at("w"));
    check(t == wt, std::string(c.key) + ": t index " +
                       (t == wt ? "EXACT" : first_diff(t, wt)));
    check(h == wh, std::string(c.key) + ": h index " +
                       (h == wh ? "EXACT" : first_diff(h, wh)));
    check(w == ww, std::string(c.key) + ": w index " +
                       (w == ww ? "EXACT" : first_diff(w, ww)));

    const int gen_t = (int)o.at("gen_t").as_int(-1);
    int mx = 0;
    for (int v : t) { mx = std::max(mx, v); }
    check(mx + 1 == gen_t,
          std::string(c.key) + ": generated tokens sit at t=" +
              std::to_string(mx + 1) + " (max(t)+1, NOT the token count " +
              std::to_string(got.size()) + ")");
  }

  std::printf("\n%d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
