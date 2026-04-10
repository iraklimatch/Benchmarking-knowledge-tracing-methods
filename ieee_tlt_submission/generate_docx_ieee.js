/**
 * Generate IEEE TLT-formatted manuscript from paper_ieee.md.
 * Usage: NODE_PATH="/Users/irakli/.npm-global/lib/node_modules" node generate_docx_ieee.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  PageNumber, PageBreak
} = require("docx");

const CONTENT_WIDTH = 9360;
const BODY = 20;   // 10pt
const SMALL = 18;  // 9pt
const TITLE_SIZE = 48; // 24pt

function p(text, opts = {}) {
  const runs = [];
  if (typeof text === "string") {
    const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
    for (const part of parts) {
      if (part.startsWith("**") && part.endsWith("**")) {
        runs.push(new TextRun({ text: part.slice(2, -2), bold: true, font: "Times New Roman", size: BODY }));
      } else if (part.startsWith("*") && part.endsWith("*")) {
        runs.push(new TextRun({ text: part.slice(1, -1), italics: true, font: "Times New Roman", size: BODY }));
      } else if (part) {
        runs.push(new TextRun({ text: part, font: "Times New Roman", size: BODY, ...opts.run }));
      }
    }
  } else {
    runs.push(...text);
  }
  return new Paragraph({ children: runs, spacing: { after: 120, line: 240 }, ...opts.para });
}

// IEEE H1: centered, small caps style, bold
function ieeeH1(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text: text.toUpperCase(), font: "Times New Roman", bold: true, size: BODY })],
  });
}

// IEEE H2: left-aligned, italic
function ieeeH2(text) {
  return new Paragraph({
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, font: "Times New Roman", italics: true, size: BODY })],
  });
}

const italic = (text) => new TextRun({ text, italics: true, font: "Times New Roman", size: BODY });
const bold = (text) => new TextRun({ text, bold: true, font: "Times New Roman", size: BODY });
const normal = (text) => new TextRun({ text, font: "Times New Roman", size: BODY });
const smallNormal = (text) => new TextRun({ text, font: "Times New Roman", size: SMALL });
const smallBold = (text) => new TextRun({ text, bold: true, font: "Times New Roman", size: SMALL });
const smallItalic = (text) => new TextRun({ text, italics: true, font: "Times New Roman", size: SMALL });

const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
const noBorder = { style: BorderStyle.NONE, size: 0 };
const topBottomBorders = { top: thinBorder, bottom: thinBorder, left: noBorder, right: noBorder };
const bottomOnlyBorder = { top: noBorder, bottom: thinBorder, left: noBorder, right: noBorder };
const noAllBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function tableCell(text, width, opts = {}) {
  const sz = opts.fontSize || SMALL;
  const runs = typeof text === "string"
    ? [new TextRun({ text, font: "Times New Roman", size: sz, ...opts.run })]
    : text.map(t => typeof t === "string" ? new TextRun({ text: t, font: "Times New Roman", size: sz }) : t);
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders: opts.borders || noAllBorders,
    margins: { top: 30, bottom: 30, left: 60, right: 60 },
    children: [new Paragraph({
      children: runs,
      alignment: opts.align || AlignmentType.LEFT,
      spacing: { after: 0, line: 220 },
    })],
  });
}

function figureImage(filename, caption, width, height) {
  const imgPath = path.join(__dirname, "figures", filename);
  if (!fs.existsSync(imgPath)) {
    console.warn(`Warning: ${imgPath} not found`);
    return [p(`[Figure: ${filename} not found]`)];
  }
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: 80 },
      children: [new ImageRun({
        type: "png", data: fs.readFileSync(imgPath),
        transformation: { width, height },
        altText: { title: caption, description: caption, name: filename },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [smallNormal(caption)],
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
  return new Table({ width: { size: CONTENT_WIDTH, type: WidthType.DXA }, columnWidths: colWidths, rows });
}

// IEEE table caption (above table): "TABLE I" + subtitle
function tableCaption(number, subtitle) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 40 },
      children: [new TextRun({ text: `TABLE ${number}`, font: "Times New Roman", bold: true, size: SMALL })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
      children: [smallItalic(subtitle)] }),
  ];
}

// ============================================================
// Read and parse paper_ieee.md
// ============================================================
const paperMd = fs.readFileSync(path.join(__dirname, "paper_ieee.md"), "utf-8");

function extractSection(md, sectionName) {
  const regex = new RegExp(`## ${sectionName}\\n([\\s\\S]*?)(?=\\n## |$)`);
  const m = md.match(regex);
  return m ? m[1].trim() : "";
}

function parseParagraphs(text) {
  return text.split(/\n\n+/).filter(s => s.trim() && !s.startsWith("**TABLE") && !s.startsWith("*") && !s.startsWith("|"));
}

function parseSubsections(sectionText) {
  return sectionText.split(/\n### /).filter(s => s.trim());
}

// ============================================================
// Build document
// ============================================================
const children = [];

// Title
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [new TextRun({
    text: "Benchmarking Knowledge Tracing Methods Across Five",
    font: "Times New Roman", size: TITLE_SIZE, bold: true,
  })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [new TextRun({
    text: "Educational Datasets: A Comparative Study of Bayesian,",
    font: "Times New Roman", size: TITLE_SIZE, bold: true,
  })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({
    text: "Logistic, and Deep Learning Approaches",
    font: "Times New Roman", size: TITLE_SIZE, bold: true,
  })],
}));

// Author removed for double-anonymous review

// Abstract
children.push(new Paragraph({
  spacing: { before: 120, after: 120 },
  children: [new TextRun({ text: "Abstract", font: "Times New Roman", bold: true, italics: true, size: BODY }),
             new TextRun({ text: "---", font: "Times New Roman", size: BODY })],
}));
const abstractText = extractSection(paperMd, "Abstract").split("\n\n")[0];
children.push(new Paragraph({
  spacing: { after: 120, line: 240 },
  indent: { left: 240, right: 240 },
  children: [new TextRun({ text: abstractText, font: "Times New Roman", italics: true, size: SMALL + 1 })],
}));

// Index Terms
children.push(new Paragraph({
  spacing: { after: 200, line: 240 },
  indent: { left: 240, right: 240 },
  children: [
    new TextRun({ text: "Index Terms", font: "Times New Roman", bold: true, italics: true, size: SMALL + 1 }),
    new TextRun({ text: "--- Knowledge tracing, educational data mining, deep learning, benchmark, learning analytics", font: "Times New Roman", size: SMALL + 1 }),
  ],
}));

// ============================================================
// Helper to process a major section with subsections
// ============================================================
function processMajorSection(sectionId, sectionTitle) {
  children.push(ieeeH1(`${sectionId}. ${sectionTitle}`));
  const text = extractSection(paperMd, `${sectionId}. ${sectionTitle.charAt(0).toUpperCase() + sectionTitle.slice(1)}`);
  if (!text) return;

  const subs = parseSubsections(text);
  for (const sub of subs) {
    const lines = sub.split("\n");
    const title = lines[0].trim();
    const body = lines.slice(1).join("\n").trim();

    if (title && !title.startsWith("#") && title.match(/^[A-G]\./)) {
      children.push(ieeeH2(title));
    }

    for (const para of parseParagraphs(body)) {
      children.push(p(para));
    }
  }
}

// ============================================================
// I. INTRODUCTION
// ============================================================
children.push(ieeeH1("I. INTRODUCTION"));
const introText = extractSection(paperMd, "I. Introduction");
for (const para of parseParagraphs(introText)) {
  // Handle numbered contribution list
  if (para.match(/^[1-4]\)/)) {
    const items = para.split(/\n/).filter(l => l.trim());
    for (const item of items) {
      children.push(new Paragraph({
        spacing: { after: 60, line: 240 },
        indent: { left: 360, hanging: 360 },
        children: [normal(item.trim())],
      }));
    }
  } else {
    children.push(p(para));
  }
}

// ============================================================
// II. RELATED WORK
// ============================================================
children.push(ieeeH1("II. RELATED WORK"));
const rwText = extractSection(paperMd, "II. Related Work");
const rwSubs = rwText.split(/\n### /).filter(s => s.trim());
for (const sub of rwSubs) {
  const lines = sub.split("\n");
  const title = lines[0].trim();
  const body = lines.slice(1).join("\n").trim();
  if (title.match(/^[A-C]\./)) children.push(ieeeH2(title));
  for (const para of parseParagraphs(body)) children.push(p(para));
}

// ============================================================
// III. METHODOLOGY
// ============================================================
children.push(ieeeH1("III. METHODOLOGY"));
const methText = extractSection(paperMd, "III. Methodology");
const methSubs = methText.split(/\n### /).filter(s => s.trim());

for (const sub of methSubs) {
  const lines = sub.split("\n");
  const title = lines[0].trim();
  const body = lines.slice(1).join("\n").trim();
  if (title.match(/^[A-E]\./)) children.push(ieeeH2(title));

  const paras = body.split(/\n\n+/);
  for (const para of paras) {
    const t = para.trim();
    if (!t) continue;

    if (t.startsWith("**TABLE I**")) {
      children.push(...tableCaption("I", "DESCRIPTIVE STATISTICS FOR THE FIVE BENCHMARK DATASETS"));
      children.push(buildTable(
        ["Dataset", "Interactions", "Students", "Items", "Skills", "Correct rate", "Avg./student"],
        [
          ["ASSISTments 2009", "278,336", "3,114", "17,708", "149", "0.659", "89.4"],
          ["ASSISTments 2015", "656,154", "14,228", "100", "100", "0.730", "46.1"],
          ["ASSISTments 2017", "934,638", "1,708", "3,162", "411", "0.374", "547.2"],
          ["Statics 2011", "189,297", "282", "1,223", "98", "0.765", "671.3"],
          ["Algebra 2005", "606,983", "567", "173,113", "271", "0.755", "1,070.5"],
        ],
        [1800, 1200, 1000, 1000, 900, 1100, 1200],
      ));
    } else if (t.startsWith("|") || t.startsWith("*")) {
      continue; // skip markdown tables/notes
    } else {
      children.push(p(t));
    }
  }
}

// ============================================================
// IV. EXPERIMENTAL RESULTS
// ============================================================
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(ieeeH1("IV. EXPERIMENTAL RESULTS"));
const resText = extractSection(paperMd, "IV. Experimental Results");
const resSubs = resText.split(/\n### /).filter(s => s.trim());

for (const sub of resSubs) {
  const lines = sub.split("\n");
  const title = lines[0].trim();
  const body = lines.slice(1).join("\n").trim();
  if (title.match(/^[A-G]\./)) children.push(ieeeH2(title));

  const paras = body.split(/\n\n+/);
  for (const para of paras) {
    const t = para.trim();
    if (!t) continue;

    if (t.startsWith("**TABLE II**")) {
      children.push(...tableCaption("II", "MEAN AUC-ROC (AND STANDARD DEVIATION) BY MODEL AND DATASET"));
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
        [1800, 1400, 1400, 1400, 1400, 1600],
        { boldRow: (ri) => ri === 5 },
      ));
    } else if (t.startsWith("**TABLE III**")) {
      children.push(...tableCaption("III", "COMPLETE BENCHMARK RESULTS: MEAN VALUES ACROSS 5-FOLD CROSS-VALIDATION"));
      children.push(buildTable(
        ["Dataset", "Model", "AUC-ROC", "PR-AUC", "Accuracy", "F1", "RMSE"],
        [
          ["ASSIST 09", "BKT", ".706", ".793", ".707", ".799", ".444"],
          ["", "PFA", ".701", ".791", ".694", ".795", ".448"],
          ["", "DKT", ".735", ".821", ".716", ".801", ".435"],
          ["", "SAKT", ".723", ".812", ".707", ".797", ".441"],
          ["", "TKT", ".728", ".815", ".712", ".801", ".438"],
          ["ASSIST 15", "BKT", ".668", ".826", ".728", ".834", ".433"],
          ["", "PFA", ".678", ".838", ".724", ".834", ".432"],
          ["", "DKT", ".717", ".863", ".731", ".835", ".423"],
          ["", "SAKT", ".702", ".853", ".728", ".834", ".427"],
          ["", "TKT", ".707", ".855", ".732", ".836", ".425"],
          ["ASSIST 17", "BKT", ".641", ".517", ".659", ".327", ".467"],
          ["", "PFA", ".631", ".499", ".650", ".257", ".469"],
          ["", "DKT", ".677", ".561", ".673", ".415", ".458"],
          ["", "SAKT", ".646", ".525", ".662", ".341", ".466"],
          ["", "TKT", ".659", ".547", ".672", ".374", ".462"],
          ["Statics 11", "BKT", ".656", ".856", ".767", ".865", ".410"],
          ["", "PFA", ".661", ".862", ".767", ".867", ".411"],
          ["", "DKT", ".698", ".880", ".770", ".868", ".404"],
          ["", "SAKT", ".676", ".869", ".767", ".867", ".408"],
          ["", "TKT", ".690", ".875", ".769", ".868", ".405"],
          ["Algebra 05", "BKT", ".753", ".902", ".799", ".879", ".382"],
          ["", "PFA", ".751", ".902", ".796", ".879", ".384"],
          ["", "DKT", ".796", ".922", ".809", ".885", ".370"],
          ["", "SAKT", ".775", ".913", ".802", ".881", ".377"],
          ["", "TKT", ".793", ".921", ".810", ".886", ".370"],
        ],
        [1400, 1200, 1100, 1100, 1100, 900, 1100],
      ));
    } else if (t.startsWith("**TABLE IV**")) {
      children.push(...tableCaption("IV", "AVERAGE TRAINING TIME PER FOLD IN SECONDS"));
      children.push(buildTable(
        ["Model", "ASSIST 09", "ASSIST 15", "ASSIST 17", "Statics 11", "Algebra 05", "Mean"],
        [
          ["BKT", "45.3", "141.5", "194.3", "24.8", "71.7", "95.5"],
          ["PFA", "3.8", "1.4", "17.3", "11.0", "38.4", "14.4"],
          ["DKT", "33.6", "79.5", "33.9", "9.2", "22.3", "35.7"],
          ["SAKT", "26.9", "64.5", "19.9", "7.3", "15.0", "26.7"],
          ["TransformerKT", "98.1", "253.3", "77.2", "28.8", "58.6", "103.2"],
        ],
        [1400, 1200, 1200, 1200, 1200, 1200, 1000],
      ));
    } else if (t.startsWith("**TABLE V**")) {
      children.push(...tableCaption("V", "95% STUDENT-CLUSTERED BOOTSTRAP CONFIDENCE INTERVALS FOR AUC-ROC"));
      children.push(buildTable(
        ["Dataset", "BKT", "PFA", "DKT", "SAKT", "TransformerKT"],
        [
          ["ASSIST 09", "[.693,.721]", "[.688,.714]", "[.722,.749]", "[.710,.737]", "[.714,.742]"],
          ["ASSIST 15", "[.662,.675]", "[.672,.685]", "[.712,.723]", "[.696,.708]", "[.701,.713]"],
          ["ASSIST 17", "[.634,.648]", "[.625,.638]", "[.670,.684]", "[.638,.654]", "[.650,.667]"],
          ["Statics 11", "[.639,.675]", "[.643,.678]", "[.684,.710]", "[.660,.691]", "[.675,.704]"],
          ["Algebra 05", "[.743,.765]", "[.742,.760]", "[.787,.805]", "[.765,.785]", "[.784,.803]"],
        ],
        [1800, 1400, 1400, 1400, 1400, 1600],
      ));
    } else if (t.startsWith("|") || t.startsWith("*Note") || t.startsWith("*")) {
      continue;
    } else if (t.match(/^Fig\. \d/)) {
      const figMap = {
        "Fig. 1": ["fig1_auc_comparison.png", "Fig. 1. AUC-ROC by model and dataset.", 500, 250],
        "Fig. 2": ["fig2_prauc_comparison.png", "Fig. 2. PR-AUC by model and dataset.", 500, 250],
        "Fig. 3": ["fig3_rmse_comparison.png", "Fig. 3. RMSE by model and dataset.", 500, 250],
        "Fig. 4": ["fig4_training_time.png", "Fig. 4. Training time per fold by model and dataset.", 500, 250],
        "Fig. 5": ["fig5_stability.png", "Fig. 5. Distribution of AUC-ROC across folds.", 550, 165],
        "Fig. 6": ["fig6_macro_average.png", "Fig. 6. Macro-averaged performance.", 450, 250],
      };
      let matched = false;
      for (const [fig, info] of Object.entries(figMap)) {
        if (t.includes(fig)) {
          children.push(p(t));
          children.push(...figureImage(info[0], info[1], info[2], info[3]));
          matched = true;
          break;
        }
      }
      if (!matched) children.push(p(t));
    } else {
      children.push(p(t));
    }
  }
}

// ============================================================
// V. DISCUSSION
// ============================================================
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(ieeeH1("V. DISCUSSION"));
const discText = extractSection(paperMd, "V. Discussion");
const discSubs = discText.split(/\n### /).filter(s => s.trim());
for (const sub of discSubs) {
  const lines = sub.split("\n");
  const title = lines[0].trim();
  const body = lines.slice(1).join("\n").trim();
  if (title.match(/^[A-D]\./)) children.push(ieeeH2(title));
  for (const para of parseParagraphs(body)) children.push(p(para));
}

// ============================================================
// VI. CONCLUSION
// ============================================================
children.push(ieeeH1("VI. CONCLUSION"));
const concText = extractSection(paperMd, "VI. Conclusion");
for (const para of parseParagraphs(concText)) children.push(p(para));

// ============================================================
// ACKNOWLEDGMENT
// ============================================================
children.push(ieeeH1("ACKNOWLEDGMENT"));
const ackText = extractSection(paperMd, "Acknowledgment");
if (ackText) children.push(p(ackText));

// ============================================================
// REFERENCES
// ============================================================
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(ieeeH1("REFERENCES"));
const refsText = extractSection(paperMd, "References");
for (const ref of refsText.split("\n\n").filter(r => r.trim())) {
  const cleaned = ref.trim().replace(/\*([^*]+)\*/g, "$1");
  children.push(new Paragraph({
    spacing: { after: 80, line: 220 },
    indent: { left: 360, hanging: 360 },
    children: [new TextRun({ text: cleaned, font: "Times New Roman", size: SMALL })],
  }));
}

// ============================================================
// AUTHOR BIOGRAPHY
// ============================================================
children.push(new Paragraph({ spacing: { before: 360 } }));
children.push(new Paragraph({
  spacing: { after: 120 },
  children: [normal("[Author biography removed for blind review.]")],
}));

// ============================================================
// Create document
// ============================================================
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman", size: BODY } } },
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
          alignment: AlignmentType.LEFT,
          children: [new TextRun({ text: "IEEE TRANSACTIONS ON LEARNING TECHNOLOGIES", font: "Times New Roman", size: 16 })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: "Times New Roman", size: SMALL })],
        })],
      }),
    },
    children,
  }],
});

const outPath = path.join(__dirname, "manuscript_ieee.docx");
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log(`IEEE TLT manuscript generated: ${outPath} (${(buffer.length / 1024).toFixed(0)} KB)`);
});
