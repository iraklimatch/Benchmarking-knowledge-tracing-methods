# JLA Submission Package

Submission materials for the Journal of Learning Analytics.

## File inventory

| File | Purpose | Blinded? |
|---|---|---|
| `manuscript_blinded.docx` | Main manuscript for submission | Yes |
| `cover_letter.md` | Cover letter to editor | No (editor-only) |
| `figures/` | All 6 figure PNGs (embedded in DOCX, also separate) | Yes |
| `generate_docx_jla.js` | Script to regenerate the blinded DOCX | -- |

## Blinding verification checklist

Before submitting, verify the DOCX contains:

- [ ] No author name anywhere in the document
- [ ] No GitHub URL containing author username (should read "[URL removed for blind review]")
- [ ] No first-person language ("we", "our") — uses "this study" / passive voice
- [ ] No acknowledgments section
- [ ] Word metadata clean (no author in File > Properties)

## Regeneration

If `paper.md` is updated, regenerate the blinded manuscript:

```bash
cd jla_submission
node generate_docx_jla.js
```

Requires Node.js and the `docx` npm package (installed in parent directory's node_modules).

## Post-acceptance restoration

After acceptance, restore in the final version:

1. Re-add author name and affiliation on title page
2. Replace "[URL removed for blind review]" with actual GitHub repository URL
3. Add acknowledgments section if applicable
4. Restore first-person language if preferred
