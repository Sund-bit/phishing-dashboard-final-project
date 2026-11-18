import { Component, OnInit, ChangeDetectorRef } from '@angular/core';

type CsvRow = Record<string, string>;
type RocCurve = { fpr: number[]; tpr: number[]; auc: number };

const COLORS = ['#a7f3d0', '#93c5fd', '#fcd34d', '#f9a8d4'];
const ORIGINAL = 'Original (16)';

@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {
  meta: Record<string, any> = {};
  main: CsvRow[] = [];
  novelty: CsvRow[] = [];
  ablation: CsvRow[] = [];
  learning: CsvRow[] = [];
  categories: CsvRow[] = [];
  featureStats: CsvRow[] = [];
  cv: CsvRow[] = [];
  ipFree: CsvRow[] = [];
  audit: CsvRow[] = [];
  robust: CsvRow[] = [];
  robustLong: CsvRow[] = [];
  robustExamples: CsvRow[] = [];
  x1: CsvRow[] = [];
  x2: CsvRow[] = [];
  x3: CsvRow[] = [];
  x4: CsvRow[] = [];
  x5: CsvRow[] = [];
  x5sig: CsvRow[] = [];
  x2meta: Record<string, any> = {};
  x2Model = 'XGBoost';
  robustModel = 'Gradient Boosting';
  roc: Record<string, RocCurve> = {};

  rocSplit = 'Stratified 80/20';
  rocFeatures = ORIGINAL;

  loading = true;
  error = '';

  readonly metrics = [
    { label: 'Accuracy', key: 'Accuracy' },
    { label: 'Precision', key: 'Precision' },
    { label: 'Recall', key: 'Recall' },
    { label: 'F1-score', key: 'F1' },
    { label: 'ROC-AUC', key: 'ROC_AUC' }
  ];

  constructor(private cdr: ChangeDetectorRef) {}

  async ngOnInit(): Promise<void> {
    try {
      const base = document.querySelector('base')?.href ?? window.location.href;
      const get = async (file: string) => {
        const r = await fetch(new URL('research-data/' + file, base));
        if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`);
        return r.text();
      };
      const [meta, main, nov, abl, lc, cat, fs, cv, roc, ipf, aud, rob, robl, robx] = await Promise.all([
        get('experiment_meta.json'),
        get('main_four_models.csv'),
        get('novelty_original_vs_enhanced.csv'),
        get('novelty_family_ablation.csv'),
        get('novelty_learning_curve.csv'),
        get('novelty_attack_category_recall.csv'),
        get('novel_feature_statistics.csv'),
        get('cv5_train_results.csv'),
        get('roc_curves.json'),
        get('novelty_ip_free_benchmark.csv'),
        get('robust_shortcut_audit.csv'),
        get('robust_summary.csv'),
        get('robust_benchmark_results.csv'),
        get('robust_benchmark_examples.csv')
      ]);
      this.meta = JSON.parse(meta);
      this.main = this.parseCsv(main);
      this.novelty = this.parseCsv(nov);
      this.ablation = this.parseCsv(abl);
      this.learning = this.parseCsv(lc);
      this.categories = this.parseCsv(cat);
      this.featureStats = this.parseCsv(fs);
      this.cv = this.parseCsv(cv);
      this.roc = JSON.parse(roc);
      this.ipFree = this.parseCsv(ipf);
      this.audit = this.parseCsv(aud);
      this.robust = this.parseCsv(rob);
      this.robustLong = this.parseCsv(robl);
      this.robustExamples = this.parseCsv(robx);
      // extended validation (optional files: the page still works without them)
      const opt = async (file: string) => {
        try {
          const r = await fetch(new URL('research-data/' + file, base));
          return r.ok ? await r.text() : '';
        } catch {
          return '';
        }
      };
      const [x1, x2, x3, x4, x5, x5s, x2m] = await Promise.all([
        opt('x1_shortcut_audit_two_datasets.csv'), opt('x2_cross_dataset_phiusiil.csv'),
        opt('x3_shap_xgboost.csv'), opt('x4_char_cnn.csv'), opt('x5_seed_summary.csv'),
        opt('x5_significance.csv'), opt('x2_meta.json')
      ]);
      this.x1 = x1 ? this.parseCsv(x1) : [];
      this.x2 = x2 ? this.parseCsv(x2) : [];
      this.x3 = x3 ? this.parseCsv(x3) : [];
      this.x4 = x4 ? this.parseCsv(x4) : [];
      this.x5 = x5 ? this.parseCsv(x5) : [];
      this.x5sig = x5s ? this.parseCsv(x5s) : [];
      this.x2meta = x2m ? JSON.parse(x2m) : {};
      if (!this.main.length) throw new Error('main_four_models.csv is empty');
    } catch (error) {
      console.error('Dashboard loading error:', error);
      this.error = 'Could not load research results. Run ml/run_experiments.py and refresh.';
    } finally {
      this.loading = false;
      this.cdr.detectChanges();
    }
  }

  private parseCsv(csv: string): CsvRow[] {
    const text = csv.replace(/^\uFEFF/, '').trim();

    if (!text) {
      return [];
    }

    const rows: string[][] = [];
    let row: string[] = [];
    let field = '';
    let inQuotes = false;

    for (let i = 0; i < text.length; i++) {
      const char = text[i];

      if (char === '"') {
        if (inQuotes && text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === ',' && !inQuotes) {
        row.push(field.trim());
        field = '';
      } else if ((char === '\n' || char === '\r') && !inQuotes) {
        if (char === '\r' && text[i + 1] === '\n') {
          i++;
        }

        row.push(field.trim());

        if (row.some(value => value !== '')) {
          rows.push(row);
        }

        row = [];
        field = '';
      } else {
        field += char;
      }
    }

    row.push(field.trim());

    if (row.some(value => value !== '')) {
      rows.push(row);
    }

    if (rows.length < 2) {
      return [];
    }

    const headers = rows[0].map(header => header.trim());

    return rows.slice(1).map(values => {
      const result: CsvRow = {};

      headers.forEach((header, index) => {
        result[header] = (values[index] ?? '').trim();
      });

      return result;
    });
  }

  // ---------- helpers used by the template
  num(row: CsvRow | undefined, key: string): number {
    const v = Number(row?.[key]);
    return Number.isFinite(v) ? v : NaN;
  }

  pct(row: CsvRow | undefined, key: string, digits = 2): string {
    const v = this.num(row, key);
    return Number.isFinite(v) ? (v * 100).toFixed(digits) + '%' : '—';
  }

  delta(row: CsvRow, key: string): string {
    const v = this.num(row, key);
    if (!Number.isFinite(v)) return '—';
    const p = Math.abs(v) < 5e-6 ? 0 : v * 100;
    return (p >= 0 ? '+' : '') + p.toFixed(3) + ' pp';
  }

  deltaClass(row: CsvRow, key: string): string {
    const v = this.num(row, key);
    if (!Number.isFinite(v) || Math.abs(v) < 5e-6) return 'delta-zero';
    return v > 0 ? 'delta-up' : 'delta-down';
  }

  bar(row: CsvRow, key: string): number {
    const v = this.num(row, key);
    return Number.isFinite(v) ? Math.max(0, Math.min(100, v * 100)) : 0;
  }

  color(i: number): string {
    return COLORS[i % COLORS.length];
  }

  get modelNames(): string[] {
    return this.main.map(r => r['Model']);
  }

  enhancedName(): string {
    return `Enhanced (16+${this.meta['n_novel_features'] ?? '?'})`;
  }

  noveltyKey(row: CsvRow, metric: string, enhanced: boolean): string {
    return `${metric}__${enhanced ? this.enhancedName() : ORIGINAL}`;
  }

  cvRow(model: string, fs = ORIGINAL): CsvRow | undefined {
    return this.cv.find(r => r['Model'] === model && r['Feature_set'] === fs);
  }

  cvText(model: string, metric: string): string {
    const r = this.cvRow(model);
    if (!r) return '—';
    return `${(this.num(r, metric + '_mean') * 100).toFixed(2)} ± ${(this.num(r, metric + '_std') * 100).toFixed(2)}`;
  }

  // ---------- learning curve
  get lcRows(): string[] {
    return [...new Set(this.learning.map(r => r['Train_rows']))]
      .sort((a, b) => Number(a) - Number(b));
  }

  lcValue(model: string, rows: string, enhanced: boolean): CsvRow | undefined {
    const fs = enhanced ? this.enhancedName() : ORIGINAL;
    return this.learning.find(r => r['Model'] === model && r['Train_rows'] === rows && r['Feature_set'] === fs);
  }

  lcDelta(model: string, rows: string): string {
    const a = this.num(this.lcValue(model, rows, false), 'F1');
    const b = this.num(this.lcValue(model, rows, true), 'F1');
    if (!Number.isFinite(a) || !Number.isFinite(b)) return '—';
    const d = (b - a) * 100;
    return (d >= 0 ? '+' : '') + d.toFixed(2);
  }

  lcDeltaClass(model: string, rows: string): string {
    const v = Number(this.lcDelta(model, rows));
    return !Number.isFinite(v) || Math.abs(v) < 0.005 ? 'delta-zero' : v > 0 ? 'delta-up' : 'delta-down';
  }

  // ---------- attack categories
  get categoryNames(): string[] {
    return [...new Set(this.categories.map(r => r['Category']))];
  }

  cat(model: string, category: string, enhanced: boolean): CsvRow | undefined {
    const fs = enhanced ? this.enhancedName() : ORIGINAL;
    return this.categories.find(r => r['Model'] === model && r['Category'] === category && r['Feature_set'] === fs);
  }

  // ---------- robustness (main novelty)
  p2(row: CsvRow | undefined, key: string, digits = 1): string {
    const v = this.num(row, key);
    return Number.isFinite(v) ? v.toFixed(digits) + '%' : '—';
  }

  get robustModels(): string[] {
    return [...new Set(this.robust.map(r => r['Model']))];
  }

  get defences(): string[] {
    return [...new Set(this.robust.map(r => r['Defence']))];
  }

  get robustTests(): string[] {
    return [...new Set(this.robustLong.map(r => r['Test']))];
  }

  robustRow(defence: string, model = this.robustModel): CsvRow | undefined {
    return this.robust.find(r => r['Defence'] === defence && r['Model'] === model);
  }

  robustRate(defence: string, test: string): number {
    const r = this.robustLong.find(x => x['Defence'] === defence && x['Model'] === this.robustModel && x['Test'] === test);
    return this.num(r, 'Rate');
  }

  /** green = good: high detection for P tests, low false alarms for L tests */
  rateColor(test: string, v: number): string {
    if (!Number.isFinite(v)) return 'transparent';
    const good = test.startsWith('P') ? v : 100 - v;
    const hue = Math.max(0, Math.min(120, (good - 50) / 50 * 120));
    return `hsla(${hue}, 70%, 45%, 0.35)`;
  }

  exampleFor(test: string): string {
    return this.robustExamples.find(r => r['Test'] === test)?.['Example_URL'] ?? '';
  }

  isProposed(defence: string): boolean {
    return defence.startsWith('D6');
  }

  // ---------- extended validation
  x1Attrs(): string[] {
    return [...new Set(this.x1.map(r => r['Attribute']))];
  }

  x1Val(dataset: string, attr: string, key: string): CsvRow | undefined {
    return this.x1.find(r => r['Dataset'] === dataset && r['Attribute'] === attr);
  }

  get x2Models(): string[] {
    return [...new Set(this.x2.map(r => r['Model']))];
  }

  get x2Defences(): string[] {
    return [...new Set(this.x2.map(r => r['Defence']))];
  }

  x2Row(defence: string, model = this.x2Model): CsvRow | undefined {
    return this.x2.find(r => r['Defence'] === defence && r['Model'] === model);
  }

  isMixed(defence: string): boolean {
    return defence.startsWith('D6');
  }

  shapGroups(defence: string, data: string): { group: string; share: number }[] {
    const rows = this.x3.filter(r => r['Defence'] === defence && r['Data'] === data);
    const by: Record<string, number> = {};
    rows.forEach(r => by[r['Group']] = (by[r['Group']] ?? 0) + this.num(r, 'Share_pct'));
    return Object.entries(by).map(([group, share]) => ({ group, share })).sort((a, b) => b.share - a.share);
  }

  shapTop(defence: string, data: string, n = 8): CsvRow[] {
    return this.x3.filter(r => r['Defence'] === defence && r['Data'] === data)
      .sort((a, b) => this.num(b, 'Share_pct') - this.num(a, 'Share_pct')).slice(0, n);
  }

  get shapDefences(): string[] {
    return [...new Set(this.x3.map(r => r['Defence']))];
  }

  groupColor(g: string): string {
    return g.startsWith('Format') ? '#f87171' : g.startsWith('Suspicious') ? '#34d399' : '#60a5fa';
  }

  x5Row(defence: string, model = this.x2Model): CsvRow | undefined {
    return this.x5.find(r => r['Defence'] === defence && r['Model'] === model);
  }

  pm(row: CsvRow | undefined, key: string, d = 1): string {
    const m = this.num(row, key + '_mean');
    const s = this.num(row, key + '_std');
    if (!Number.isFinite(m)) return '—';
    return `${m.toFixed(d)} ± ${Number.isFinite(s) ? s.toFixed(d) : '0'}`;
  }

  pval(row: CsvRow): string {
    const p = this.num(row, 'Wilcoxon_p_value');
    if (!Number.isFinite(p)) return '—';
    return p < 0.001 ? '< 0.001' : p.toFixed(3);
  }

  ipFreeModels(): string[] {
    return [...new Set(this.ipFree.map(r => r['Model']))];
  }

  ipf(model: string, split: string, enhanced: boolean): CsvRow | undefined {
    const fs = enhanced ? this.enhancedName() : ORIGINAL;
    return this.ipFree.find(r => r['Model'] === model && r['Split'] === split && r['Feature_set'] === fs);
  }

  errReduction(model: string, split: string): string {
    const a = this.num(this.ipf(model, split, false), 'Errors_mean');
    const b = this.num(this.ipf(model, split, true), 'Errors_mean');
    if (!Number.isFinite(a) || !Number.isFinite(b) || a === 0) return '—';
    const r = (a - b) / a * 100;
    return (r >= 0 ? '−' : '+') + Math.abs(r).toFixed(1) + '%';
  }

  errClass(model: string, split: string): string {
    const a = this.num(this.ipf(model, split, false), 'Errors_mean');
    const b = this.num(this.ipf(model, split, true), 'Errors_mean');
    return b < a ? 'delta-up' : b > a ? 'delta-down' : 'delta-zero';
  }

  ablationFor(split: string): CsvRow[] {
    return this.ablation.filter(r => r['Split'] === split);
  }

  // ---------- ROC curve as SVG path
  rocPath(model: string): string {
    const c = this.roc[`${model} | ${this.rocFeatures} | ${this.rocSplit}`];
    if (!c) return '';
    // zoom into the top-left corner (FPR 0-0.02, TPR 0.98-1) where the models differ
    const x = (f: number) => 40 + Math.min(f, 0.02) / 0.02 * 440;
    const y = (t: number) => 20 + (1 - Math.max(t, 0.98)) / 0.02 * 300;
    return c.fpr.map((f, i) => `${i ? 'L' : 'M'}${x(f).toFixed(1)},${y(c.tpr[i]).toFixed(1)}`).join(' ');
  }

  rocAuc(model: string): string {
    const c = this.roc[`${model} | ${this.rocFeatures} | ${this.rocSplit}`];
    return c ? c.auc.toFixed(5) : '—';
  }

  setRoc(split: string, features: string): void {
    this.rocSplit = split;
    this.rocFeatures = features;
  }
}
