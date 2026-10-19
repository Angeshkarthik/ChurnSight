# ChurnSight — Frontend Application

Modern React 19 + TypeScript frontend dashboard for **ChurnSight** (*"See risk. Understand why."*), providing real-time customer churn scoring, operational risk tiering, and local Explainable AI (SHAP) feature attributions.

---

## 🛠️ Architecture & Tech Stack

- **Framework:** [React 19](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- **Bundler & Tooling:** [Vite 8](https://vite.dev/)
- **Design System:** Custom Vanilla CSS enterprise SaaS light theme (Charcoal `#111827` + Emerald `#10B981`)
- **Icons:** [Lucide React](https://lucide.dev/)
- **Testing:** [Vitest](https://vitest.dev/) + [Testing Library](https://testing-library.com/) + JSDOM

---

## 🚀 Getting Started

### 1. Prerequisites
- Node.js `v18+` (tested on Node `v24.x`)
- npm `v9+`
- Running FastAPI backend service (default: `http://127.0.0.1:8000`)

### 2. Install Dependencies
```bash
cd frontend
npm install
```

### 3. Start Development Server
```bash
npm run dev
```
The application will launch locally at `http://localhost:5173/`.

### 4. Run Automated Tests
```bash
npm test
```
Executes 9 comprehensive Vitest integration and component test suites validating form inputs, numerical range rules, probability badges, and local SHAP waterfall rendering.

### 5. Build for Production
```bash
npm run build
```
Executes TypeScript type checking (`tsc -b`) and Vite production bundle compilation to `dist/`.

---

## 🔌 API Client & Backend Integration

The frontend communicates with the FastAPI backend via a centralized typed client located at `src/api/client.ts`.

### Configurable Base URL
The API endpoint is resolved via the Vite environment variable `VITE_API_BASE_URL`:

```ts
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
```

To configure a custom backend endpoint for staging or production:
1. Create a `.env` file inside `frontend/` (or set environment variables in your deployment host):
   ```env
   VITE_API_BASE_URL=http://your-backend-host:8000
   ```
2. Re-run `npm run dev` or `npm run build`.

### Endpoints Consumed
- `GET /health` — Service readiness check, model family identification, and operational threshold indicator.
- `POST /predict` — Scores individual customer features across 19 predictors against the frozen Gradient Boosting pipeline and applies the operational cutoff ($P \ge 0.35$).
- `POST /explain` — Computes local SHAP attributions on the log-odds margin scale ($F(x) = \ln(p / (1-p))$), returning signed feature contributions and natural-language narrative reasons.

---

## 📁 Frontend Directory Structure

```text
frontend/
├── index.html                           # Single-page application entry point
├── package.json                         # Dependencies & scripts
├── tsconfig.json                        # TypeScript root project references
├── tsconfig.app.json                    # Application compiler configuration
├── vite.config.ts                       # Vite & Vitest configuration
└── src/
    ├── main.tsx                         # React application DOM mounter
    ├── App.tsx                          # Root container & page view router
    ├── index.css                        # Vanilla CSS design tokens & utilities
    ├── api/
    │   └── client.ts                    # Centralized typed HTTP client
    ├── components/
    │   ├── common/                      # Reusable UI (Button, Card, Badge, Modal, BrandWordmark)
    │   ├── layout/                      # Application shell (AppShell, Sidebar, TopNav)
    │   └── analysis/                    # CustomerForm, PredictionCard, ShapWaterfall
    ├── context/
    │   └── AnalysisContext.tsx          # React Context managing session state & API orchestration
    ├── data/
    │   ├── modelArtifacts.ts            # Holdout evaluation metrics & OOF threshold curves
    │   └── sampleCustomers.ts           # Synthetic reference profiles (High, Borderline, Low risk)
    ├── pages/
    │   ├── DashboardPage.tsx            # KPI metrics, welcome banner, session history
    │   ├── CustomerAnalysisPage.tsx     # 19-feature input form, scoring & SHAP breakdown
    │   ├── ChurnPredictionsPage.tsx     # Session scoring history audit table
    │   ├── ModelInsightsPage.tsx        # Offline holdout benchmarks & threshold sweep
    │   ├── ReportsPage.tsx              # Risk distribution summary & CSV/JSON export
    │   └── SettingsPage.tsx             # Backend connectivity check & API docs links
    ├── test/
    │   ├── setup.ts                     # Testing Library DOM matchers
    │   └── app.test.tsx                 # 9 unit & integration tests
    └── types/
        └── api.ts                       # Pydantic-aligned TypeScript interfaces
```
