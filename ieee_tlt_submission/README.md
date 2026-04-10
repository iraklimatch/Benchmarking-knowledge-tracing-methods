# IEEE TLT Submission Package

Submission materials for IEEE Transactions on Learning Technologies.

## File inventory

| File | Purpose |
|---|---|
| `paper_ieee.md` | Full manuscript in IEEE TLT format (source) |
| `manuscript_ieee.docx` | Generated Word document for submission |
| `cover_letter.md` | Cover letter to editor |
| `figures/` | 6 PNG figures |
| `generate_docx_ieee.js` | DOCX generator with IEEE formatting |

## IEEE TLT format checklist

- [x] IEEE numbered citations [1] ordered by first appearance
- [x] Abstract 150-250 words
- [x] Index Terms (not "Keywords")
- [x] Separate Related Work section
- [x] Introduction ends with contributions list and roadmap
- [x] TABLE I-V with Roman numerals, captions above
- [x] Fig. 1-6 with abbreviated captions below
- [x] Conclusion section
- [x] Acknowledgment with conflict of interest statement
- [x] Author biography (placeholder - author to fill)
- [x] No AI disclosure
- [x] Single-anonymous (author name included)

## Regeneration

```bash
cd ieee_tlt_submission
NODE_PATH="/Users/irakli/.npm-global/lib/node_modules" node generate_docx_ieee.js
```

## Notes

- The DOCX is single-column (the `docx` npm library does not support double-column). For camera-ready, use the official IEEE LaTeX template (`IEEEtran.cls`).
- Author biography is a placeholder and must be filled before submission.
- References [14] and [15] (Liu et al. 2022, Schmucker et al. 2024) have been added with best-available citation information.
