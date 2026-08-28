# Contributing

These are study notes on a specific book. That shapes what's useful to contribute.

## Most valuable

**Corrections.** If a number, formula, method name, or claim here does not match the book, that's a real bug. Open an issue or PR with the book page number so it can be checked against the source. Fidelity to the book matters more than anything else in this repo.

**Broken diagrams.** Mermaid rendering differs subtly across GitHub, VS Code, and mermaid.live. If a diagram renders wrong or illegibly in dark mode, say where you saw it.

**Better diagrams.** A diagram that shows the actual mechanism beats one that shows boxes with arrows. If you can replace a weak one, do.

**Missing cross-links.** The chapters reference each other constantly. If Chapter 19 discusses something Chapter 12 derives and doesn't link to it, that's worth fixing.

## Please don't

**Don't add content that isn't in the book.** This repo summarizes one source. Your own knowledge of a newer paper is valuable, but it belongs somewhere else — mixing it in makes the notes untrustworthy, because a reader can no longer tell what's Roitman and what isn't.

**Don't reproduce the book.** These are summaries. Long verbatim passages, the full quiz bank, or wholesale copied tables would make this a republication rather than study notes, which the CC BY-SA license permits but the spirit of the project does not.

**Don't reformat wholesale.** Every chapter follows the same skeleton on purpose. If you think the skeleton is wrong, open an issue first.

## House style

Read an existing chapter before writing. The conventions:

- Chapter files follow a fixed skeleton: epigraph → metadata → nav → what it's about → TOC → mental model → pillars → key formulas → decision guide → pitfalls → summary → checklist → going deeper → nav → attribution.
- **Mermaid only** for diagrams. No PNGs, no ASCII art, no external image hosts.
- Every `classDef` sets `fill`, `stroke`, **and** `color` so labels stay legible in dark mode.
- Quote any Mermaid label containing `()`, `[]`, `{}`, `:`, `,`, or `/`. Use `\n` for line breaks, never `<br>`.
- Every diagram gets an italic caption and is referenced in nearby prose.
- Summary bullets are claims, not topics.
- Concrete numbers go in tables. They're the point.
- Expand acronyms on first use in each file — people land here from search.
- Relative links only, plain filenames: link text in square brackets, then the bare `.md` filename in parentheses — no `./` prefix, no directory. Don't link to anchors in other files.

## Before you open a PR

- [ ] Every Mermaid block renders (paste into [mermaid.live](https://mermaid.live) or preview in VS Code)
- [ ] Every relative link resolves
- [ ] Every in-file TOC anchor matches a heading
- [ ] Any number you added appears in the book, and you can name the page
