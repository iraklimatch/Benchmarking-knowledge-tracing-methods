/**
 * Generate blinded JLA submission manuscript from paper.md.
 *
 * Usage:  node generate_docx_jla.js
 * Output: manuscript_blinded.docx
 *
 * Reads ../paper.md as single source of truth, applies blinding
 * transformations, adds JLA-required sections (Notes for Practice,
 * Declaration of Conflicting Interest, Funding), and shortens abstract
 * to <= 200 words.
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat
} = require("docx");

const CONTENT_WIDTH = 9360;

// ============================================================
// Helpers (same as base generator)
// ============================================================

function p(text, opts = {}) {
  const runs = [];
  if (typeof text === "string") {
    const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
    for (const part of parts) {
      if (part.startsWith("**") && part.endsWith("**")) {
        runs.push(new TextRun({ text: part.slice(2, -2), bold: true, font: "Times New Roman", size: 24 }));
      } else if (part.startsWith("*") && part.endsWith("*")) {
        runs.push(new TextRun({ text: part.slice(1, -1), italics: true, font: "Times New Roman", size: 24 }));
      } else if (part) {
        runs.push(new TextRun({ text: part, font: "Times New Roman", size: 24, ...opts.run }));
      }
    }
  } else {
    runs.push(...text);
  }
  return new Paragraph({ children: runs, spacing: { after: 200, line: 480 }, ...opts.para });
}

function heading(text, level) {
  const sizes = { [HeadingLevel.HEADING_1]: 28, [HeadingLevel.HEADING_2]: 26, [HeadingLevel.HEADING_3]: 24 };
  return new Paragraph({
    heading: level,
    children: [new TextRun({ text, font: "Times New Roman", bold: true, size: sizes[level] || 24 })],
    spacing: { before: 360, after: 200 },
  });
}

const italic = (text) => new TextRun({ text, italics: true, font: "Times New Roman", size: 24 });
const bold = (text) => new TextRun({ text, bold: true, font: "Times New Roman", size: 24 });
const normal = (text) => new TextRun({ text, font: "Times New Roman", size: 24 });
const boldItalic = (text) => new TextRun({ text, bold: true, italics: true, font: "Times New Roman", size: 24 });

const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
const noBorder = { style: BorderStyle.NONE, size: 0 };
const topBottomBorders = { top: thinBorder, bottom: thinBorder, left: noBorder, right: noBorder };
const bottomOnlyBorder = { top: noBorder, bottom: thinBorder, left: noBorder, right: noBorder };
const noAllBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function tableCell(text, width, opts = {}) {
  const runs = typeof text === "string"
    ? [new TextRun({ text, font: "Times New Roman", size: 20, ...opts.run })]
    : text.map(t => typeof t === "string" ? new TextRun({ text: t, font: "Times New Roman", size: 20 }) : t);
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders: opts.borders || noAllBorders,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({
      children: runs,
      alignment: opts.align || AlignmentType.LEFT,
      spacing: { after: 0, line: 240 },
    })],
  });
}

function figureImage(filename, caption, width, height) {
  const imgPath = path.join(__dirname, "figures", filename);
  if (!fs.existsSync(imgPath)) {
    console.warn(`Warning: ${imgPath} not found, skipping`);
    return [p(`[Figure: ${filename} not found]`)];
  }
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 120 },
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(imgPath),
        transformation: { width, height },
        altText: { title: caption, description: caption, name: filename },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { after: 360 },
      children: [italic(caption)],
    }),
  ];
}

function buildTable(headers, data, colWidths, opts = {}) {
  const rows = [];
  rows.push(new TableRow({
    children: headers.map((h, i) => tableCell(h, colWidths[i], {
      borders: topBottomBorders, run: { bold: true },
      align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
    })),
  }));
  data.forEach((row, ri) => {
    const isLast = ri === data.length - 1;
    rows.push(new TableRow({
      children: row.map((cell, i) => tableCell(cell, colWidths[i], {
        borders: isLast ? bottomOnlyBorder : noAllBorders,
        align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        run: opts.boldRow && opts.boldRow(ri) ? { bold: true } : {},
      })),
    }));
  });
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: colWidths,
    rows,
  });
}

// ============================================================
// Read paper.md, apply blinding
// ============================================================
const rawMd = fs.readFileSync(path.join(__dirname, "..", "paper.md"), "utf-8");

// Blinding transformations
let paperMd = rawMd;
paperMd = paperMd.replace(
  /https:\/\/github\.com\/iraklimatch\/Benchmarking-knowledge-tracing-methods/g,
  "[URL removed for blind review]"
);
paperMd = paperMd.replace(
  /we include a causal Transformer/g,
  "this study includes a causal Transformer"
);
paperMd = paperMd.replace(
  /We note that this model/g,
  "This model"
);
paperMd = paperMd.replace(
  /several of our datasets/g,
  "several of the benchmark datasets"
);
paperMd = paperMd.replace(
  /and we report means/g,
  "and means"
);

function extractSection(md, sectionName) {
  const regex = new RegExp(`## ${sectionName}\\n([\\s\\S]*?)(?=\\n## |$)`);
  const match = md.match(regex);
  return match ? match[1].trim() : "";
}

function parseParagraphs(text) {
  return text.split(/\n\n+/).filter(s => s.trim() && !s.startsWith("**Table") && !s.startsWith("*") && !s.startsWith("|"));
}

// ============================================================
// JLA shortened abstract (<= 200 words)
// ============================================================
const JLA_ABSTRACT = `Knowledge tracing (KT) models student knowledge over time to predict future performance. This study benchmarks five KT methods spanning three methodological generations: Bayesian Knowledge Tracing (BKT), Performance Factors Analysis (PFA), Deep Knowledge Tracing (DKT), Self-Attentive Knowledge Tracing (SAKT), and a generic causal Transformer baseline (TransformerKT). All models were evaluated on five public datasets (ASSISTments 2009, ASSISTments 2015, ASSISTments 2017, Statics 2011, Algebra 2005) using 5-fold student-level cross-validation, five metrics (AUC-ROC, PR-AUC, accuracy, F1, RMSE), and a common prediction mask ensuring identical evaluation subsets. DKT achieved the highest macro-averaged AUC-ROC (0.725), outperforming BKT and PFA (both 0.685) by approximately 4 percentage points. TransformerKT (0.715) closely trailed DKT at 2.9 times the computational cost. Classical methods remained competitive on datasets with long interaction histories and offered substantially lower training times. Student-clustered bootstrap confidence intervals confirmed that DKT's advantage over classical methods was statistically meaningful on most datasets. These findings provide practitioners with empirically grounded guidance for model selection in learning analytics applications.`;

// ============================================================
// Build document content
// ============================================================
const children = [];

// Title (no author — blinded)
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({
    text: "Benchmarking knowledge tracing methods across five educational datasets:",
    font: "Times New Roman", size: 32, bold: true,
  })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 600 },
  children: [new TextRun({
    text: "A comparative study of Bayesian, logistic, and deep learning approaches",
    font: "Times New Roman", size: 32, bold: true,
  })],
}));

// Abstract (shortened for JLA)
children.push(heading("Abstract", HeadingLevel.HEADING_1));
children.push(p(JLA_ABSTRACT));
children.push(new Paragraph({
  spacing: { after: 200, line: 480 },
  children: [boldItalic("Keywords: "), italic("knowledge tracing, educational data mining, deep learning, benchmark, Bayesian knowledge tracing, self-attention, learning analytics")],
}));

// Introduction
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(heading("Introduction", HeadingLevel.HEADING_1));
const introText = extractSection(paperMd, "Introduction");
for (const para of parseParagraphs(introText)) {
  children.push(p(para));
}

// Method
children.push(heading("Method", HeadingLevel.HEADING_1));
const methodText = extractSection(paperMd, "Method");
const methodSections = methodText.split(/\n### /);

for (const sec of methodSections) {
  if (!sec.trim()) continue;
  const lines = sec.split("\n");
  const title = lines[0].trim();
  const body = lines.slice(1).join("\n").trim();

  if (title && !title.startsWith("#")) {
    children.push(heading(title, HeadingLevel.HEADING_2));
  }

  for (const para of parseParagraphs(body)) {
    if (para.startsWith("**Table 1**")) {
      children.push(new Paragraph({ spacing: { before: 240, after: 80 }, children: [bold("Table 1")] }));
      children.push(new Paragraph({ spacing: { after: 120 }, children: [italic("Descriptive statistics for the five benchmark datasets")] }));
      const t1Widths = [1800, 1200, 1100, 1100, 900, 1100, 1200];
      children.push(buildTable(
        ["Dataset", "Interactions", "Students", "Items", "Skills", "Correct rate", "Avg. per student"],
        [
          ["ASSISTments 2009", "278,336", "3,114", "17,708", "149", "0.659", "89.4"],
          ["ASSISTments 2015", "656,154", "14,228", "100", "100", "0.730", "46.1"],
          ["ASSISTments 2017", "934,638", "1,708", "3,162", "411", "0.374", "547.2"],
          ["Statics 2011", "189,297", "282", "1,223", "98", "0.765", "671.3"],
          ["Algebra 2005", "606,983", "567", "173,113", "271", "0.755", "1,070.5"],
        ],
        t1Widths,
      ));
    } else {
      children.push(p(para));
    }
  }
}

// Results
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(heading("Results", HeadingLevel.HEADING_1));

const resultsText = extractSection(paperMd, "Results");
const resultsSections = resultsText.split(/\n### /);

for (const sec of resultsSections) {
  if (!sec.trim()) continue;
  const lines = sec.split("\n");
  const title = lines[0].trim();
  const body = lines.slice(1).join("\n").trim();

  if (title && !title.startsWith("#")) {
    children.push(heading(title, HeadingLevel.HEADING_2));
  }

  const paras = body.split(/\n\n+/);
  for (const para of paras) {
    const trimmed = para.trim();
    if (!trimmed) continue;

    if (trimmed.startsWith("**Table 2**")) {
      children.push(new Paragraph({ spacing: { before: 240, after: 80 }, children: [bold("Table 2")] }));
      children.push(new Paragraph({ spacing: { after: 120 }, children: [italic("Mean AUC-ROC (and standard deviation) by model and dataset")] }));
      const t2Widths = [1800, 1400, 1400, 1400, 1400, 1600];
      children.push(buildTable(
        ["Dataset", "BKT", "PFA", "DKT", "SAKT", "TransformerKT"],
        [
          ["ASSISTments 2009", ".706 (.005)", ".701 (.005)", ".735 (.006)", ".723 (.006)", ".728 (.006)"],
          ["ASSISTments 2015", ".668 (.002)", ".678 (.001)", ".717 (.001)", ".702 (.002)", ".707 (.002)"],
          ["ASSISTments 2017", ".641 (.003)", ".631 (.004)", ".677 (.003)", ".646 (.003)", ".659 (.004)"],
          ["Statics 2011", ".656 (.006)", ".661 (.007)", ".698 (.011)", ".676 (.009)", ".690 (.010)"],
          ["Algebra 2005", ".753 (.007)", ".751 (.008)", ".796 (.006)", ".775 (.007)", ".793 (.007)"],
          ["Macro avg.", ".685", ".685", ".725", ".704", ".715"],
        ],
        t2Widths,
        { boldRow: (ri) => ri === 5 },
      ));
    } else if (trimmed.startsWith("**Table 3**")) {
      children.push(new Paragraph({ spacing: { before: 240, after: 80 }, children: [bold("Table 3")] }));
      children.push(new Paragraph({ spacing: { after: 120 }, children: [italic("Complete benchmark results: mean values across 5-fold cross-validation")] }));
      const t3Widths = [1600, 1400, 1100, 1100, 1100, 900, 1100];
      children.push(buildTable(
        ["Dataset", "Model", "AUC-ROC", "PR-AUC", "Accuracy", "F1", "RMSE"],
        [
          ["ASSISTments 2009", "BKT", ".706", ".793", ".707", ".799", ".444"],
          ["", "PFA", ".701", ".791", ".694", ".795", ".448"],
          ["", "DKT", ".735", ".821", ".716", ".801", ".435"],
          ["", "SAKT", ".723", ".812", ".707", ".797", ".441"],
          ["", "TransformerKT", ".728", ".815", ".712", ".801", ".438"],
          ["ASSISTments 2015", "BKT", ".668", ".826", ".728", ".834", ".433"],
          ["", "PFA", ".678", ".838", ".724", ".834", ".432"],
          ["", "DKT", ".717", ".863", ".731", ".835", ".423"],
          ["", "SAKT", ".702", ".853", ".728", ".834", ".427"],
          ["", "TransformerKT", ".707", ".855", ".732", ".836", ".425"],
          ["ASSISTments 2017", "BKT", ".641", ".517", ".659", ".327", ".467"],
          ["", "PFA", ".631", ".499", ".650", ".257", ".469"],
          ["", "DKT", ".677", ".561", ".673", ".415", ".458"],
          ["", "SAKT", ".646", ".525", ".662", ".341", ".466"],
          ["", "TransformerKT", ".659", ".547", ".672", ".374", ".462"],
          ["Statics 2011", "BKT", ".656", ".856", ".767", ".865", ".410"],
          ["", "PFA", ".661", ".862", ".767", ".867", ".411"],
          ["", "DKT", ".698", ".880", ".770", ".868", ".404"],
          ["", "SAKT", ".676", ".869", ".767", ".867", ".408"],
          ["", "TransformerKT", ".690", ".875", ".769", ".868", ".405"],
          ["Algebra 2005", "BKT", ".753", ".902", ".799", ".879", ".382"],
          ["", "PFA", ".751", ".902", ".796", ".879", ".384"],
          ["", "DKT", ".796", ".922", ".809", ".885", ".370"],
          ["", "SAKT", ".775", ".913", ".802", ".881", ".377"],
          ["", "TransformerKT", ".793", ".921", ".810", ".886", ".370"],
        ],
        t3Widths,
      ));
    } else if (trimmed.startsWith("**Table 4**")) {
      children.push(new Paragraph({ spacing: { before: 240, after: 80 }, children: [bold("Table 4")] }));
      children.push(new Paragraph({ spacing: { after: 120 }, children: [italic("Average training time per fold in seconds")] }));
      const t4Widths = [1400, 1200, 1200, 1200, 1200, 1200, 1000];
      children.push(buildTable(
        ["Model", "ASSIST 09", "ASSIST 15", "ASSIST 17", "Statics 11", "Algebra 05", "Mean"],
        [
          ["BKT", "45.3", "141.5", "194.3", "24.8", "71.7", "95.5"],
          ["PFA", "3.8", "1.4", "17.3", "11.0", "38.4", "14.4"],
          ["DKT", "33.6", "79.5", "33.9", "9.2", "22.3", "35.7"],
          ["SAKT", "26.9", "64.5", "19.9", "7.3", "15.0", "26.7"],
          ["TransformerKT", "98.1", "253.3", "77.2", "28.8", "58.6", "103.2"],
        ],
        t4Widths,
      ));
    } else if (trimmed.startsWith("**Table 5**")) {
      children.push(new Paragraph({ spacing: { before: 240, after: 80 }, children: [bold("Table 5")] }));
      children.push(new Paragraph({ spacing: { after: 120 }, children: [italic("95% student-clustered bootstrap confidence intervals for AUC-ROC (mean across 5 folds)")] }));
      const t5Widths = [1800, 1400, 1400, 1400, 1400, 1600];
      children.push(buildTable(
        ["Dataset", "BKT", "PFA", "DKT", "SAKT", "TransformerKT"],
        [
          ["ASSISTments 2009", "[.693, .721]", "[.688, .714]", "[.722, .749]", "[.710, .737]", "[.714, .742]"],
          ["ASSISTments 2015", "[.662, .675]", "[.672, .685]", "[.712, .723]", "[.696, .708]", "[.701, .713]"],
          ["ASSISTments 2017", "[.634, .648]", "[.625, .638]", "[.670, .684]", "[.638, .654]", "[.650, .667]"],
          ["Statics 2011", "[.639, .675]", "[.643, .678]", "[.684, .710]", "[.660, .691]", "[.675, .704]"],
          ["Algebra 2005", "[.743, .765]", "[.742, .760]", "[.787, .805]", "[.765, .785]", "[.784, .803]"],
        ],
        t5Widths,
      ));
    } else if (trimmed.startsWith("|") || trimmed.startsWith("*Note")) {
      continue;
    } else if (trimmed.startsWith("Figure ")) {
      const figMap = {
        "Figure 1": ["fig1_auc_comparison.png", "Figure 1. Area under the ROC curve by model and dataset", 550, 275],
        "Figure 2": ["fig2_prauc_comparison.png", "Figure 2. Area under the precision-recall curve by model and dataset", 550, 275],
        "Figure 3": ["fig3_rmse_comparison.png", "Figure 3. Root mean square error by model and dataset", 550, 275],
        "Figure 4": ["fig4_training_time.png", "Figure 4. Average training time per fold by model and dataset", 550, 275],
        "Figure 5": ["fig5_stability.png", "Figure 5. Distribution of AUC-ROC scores across folds for each dataset", 600, 180],
        "Figure 6": ["fig6_macro_average.png", "Figure 6. Macro-averaged performance across all five datasets", 500, 275],
      };
      for (const [fig, info] of Object.entries(figMap)) {
        if (trimmed.includes(fig + " ") || trimmed.includes(fig + ".")) {
          children.push(p(trimmed));
          children.push(...figureImage(info[0], info[1], info[2], info[3]));
          break;
        }
      }
      if (!Object.keys(figMap).some(fig => trimmed.includes(fig + " ") || trimmed.includes(fig + "."))) {
        children.push(p(trimmed));
      }
    } else {
      children.push(p(trimmed));
    }
  }
}

// Discussion
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(heading("Discussion", HeadingLevel.HEADING_1));

const discussionText = extractSection(paperMd, "Discussion");
const discussionSections = discussionText.split(/\n### /);

for (const sec of discussionSections) {
  if (!sec.trim()) continue;
  const lines = sec.split("\n");
  const title = lines[0].trim();
  const body = lines.slice(1).join("\n").trim();

  if (title && !title.startsWith("#")) {
    children.push(heading(title, HeadingLevel.HEADING_2));
  }
  for (const para of parseParagraphs(body)) {
    children.push(p(para));
  }
}

// ============================================================
// JLA-required: Notes for Practice
// ============================================================
children.push(heading("Notes for Practice", HeadingLevel.HEADING_1));

const practiceNotes = [
  "DKT provides the best accuracy-to-cost tradeoff for learning analytics deployments requiring high predictive performance, achieving the highest AUC-ROC on all five datasets.",
  "BKT and PFA remain viable choices when interpretability is required or computational resources are limited, particularly on datasets with long interaction histories where BKT achieves AUC-ROC within 4 percentage points of DKT.",
  "Attention-based models (SAKT, TransformerKT) do not provide sufficient gains over DKT to justify their increased complexity in most practical settings. TransformerKT requires 2.9 times more computation for 0.9 AUC percentage points less than DKT.",
  "A common prediction mask that excludes first skill encounters and sequence boundaries should be adopted as standard evaluation practice in knowledge tracing to ensure fair cross-method comparisons.",
  "Student-clustered bootstrap confidence intervals are recommended over interaction-level resampling to correctly quantify uncertainty in knowledge tracing benchmark results, as within-student correlation can produce misleadingly narrow intervals.",
];

for (const note of practiceNotes) {
  children.push(new Paragraph({
    spacing: { after: 160, line: 480 },
    indent: { left: 360, hanging: 360 },
    children: [
      new TextRun({ text: "\u2022  ", font: "Times New Roman", size: 24 }),
      new TextRun({ text: note, font: "Times New Roman", size: 24 }),
    ],
  }));
}

// References
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(heading("References", HeadingLevel.HEADING_1));

const refsText = extractSection(paperMd, "References");
for (const ref of refsText.split("\n\n").filter(r => r.trim())) {
  children.push(new Paragraph({
    spacing: { after: 120, line: 480 },
    indent: { left: 720, hanging: 720 },
    children: [normal(ref.trim().replace(/\*([^*]+)\*/g, "$1"))],
  }));
}

// ============================================================
// JLA-required: Declaration of Conflicting Interest
// ============================================================
children.push(heading("Declaration of Conflicting Interest", HeadingLevel.HEADING_1));
children.push(p("The author(s) declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article."));

// ============================================================
// JLA-required: Funding
// ============================================================
children.push(heading("Funding", HeadingLevel.HEADING_1));
children.push(p("This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors."));

// ============================================================
// Create document (no author metadata for blinding)
// ============================================================
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "BENCHMARKING KNOWLEDGE TRACING", font: "Times New Roman", size: 20 })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: "Times New Roman", size: 20 })],
        })],
      }),
    },
    children,
  }],
});

const outPath = path.join(__dirname, "manuscript_blinded.docx");
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log(`JLA blinded manuscript generated: ${outPath} (${(buffer.length / 1024).toFixed(0)} KB)`);
});
