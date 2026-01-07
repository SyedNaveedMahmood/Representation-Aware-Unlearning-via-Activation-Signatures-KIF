
# Repository Guidelines
## PHILOSOPHY
Think like a senior engineer, code like a craftsman, explain like a teacher.  
Understand before acting; optimize for maintainability; respect time; never assume; explain clearly; test rigorously.

## CORE PRINCIPLES
- Investigate first, act second: read relevant files, understand flow, then fix.
- Minimize token usage: read in chunks, use regex search, parallelize independent reads.
- No hallucinations: verify functions/imports, read code before referencing, test builds.

## Collaboration Rules for This Repo
- Confirm Understanding with the user. **For every request the user gives, Reply with your understanding to the user**. This way there will be proper understanding with the user and the work done. 
- The assistant must **only perform actions explicitly requested by the user**. No proactive refactors, design changes, or “improvements” unless the user directly asks for them.
- If a task would require an assumption (about behavior, UX, data, naming, etc.), the assistant must **stop and ask clarifying questions or offer concrete options for the user to choose from** before making changes.
- The assistant must **avoid changing existing visual design** (spacing, colors, layout, animations) except where the user has requested a specific change and agreed to the concrete implementation approach.
- When there are multiple reasonable implementation options, the assistant should **present the options briefly and wait for the user’s explicit choice** before editing code.
- All behavior changes should be **as small and targeted as possible**, touching only the files and logic directly related to the user’s request.
- Communication: stay concise, surface options/tradeoffs quickly, and propose pragmatic next steps when prompted; keep docs in sync with changes to avoid re-parsing the codebase.

## WORKFLOW
1. Ask clarifying questions (if needed)  
2. Search (grep/semantic)  
3. Read code to understand implementation  
4. Identify root cause  
5. Make minimal fix  
6. `npm run build` (verify)  
7. Explain what was fixed

## SMART RESPONSES
- Explain simply (e.g., “staleTime is like milk expiration”).
- Ask questions before assuming (“what’s happening exactly?”).

## Project Structure & Module Organization
- Frontend (Vite + React + TypeScript) lives at `src/` with feature directories such as `pages/` (route-level screens), `components/` (reusable UI, layout, and guards), `state/` (Zustand stores), `api/` (Axios clients and query helpers), `hooks/`, `layout/`, and `styles/`.
- Mock/data scaffolding sits in `src/mocks/` and shared types in `src/types/`.
- Backend API is in `server/` (Express + Mongoose + Supabase client). Routes sit under `server/routes/`, data models under `server/models/`, and utilities under `server/utils/`.
- Public assets are in `public/`; build output goes to `dist/` (ignored).

## Build, Test, and Development Commands
- Install deps: `npm install` (root) and `npm install` inside `server/` for API dependencies.
- Run frontend locally: `npm run dev` (Vite dev server).
- Build frontend: `npm run build` (type-check via `tsc` then Vite bundle).
- Preview built frontend: `npm run preview`.
- Lint frontend: `npm run lint` (ESLint flat config).
- Format: `npm run format` (Prettier).
- Run API: from `server/`, `npm start` (requires `.env` with `MONGODB_URI`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`).
- Seed data: from `server/`, `npm run seed` (populates MongoDB using local seed script).

## Coding Style & Naming Conventions
- TypeScript-first; keep components in PascalCase (`ProjectShell.tsx`, `DashboardPage.tsx`), hooks prefixed with `use`, Zustand stores in camelCase (`authStore.ts`).
- Prefer functional, memo-friendly React patterns and colocate component-specific styles/logic near the component.
- Tailwind is configured via `tailwind.config.js` and global styles live in `src/styles/globals.css`; favor utility classes over ad-hoc CSS.
- Lint with ESLint config in `eslint.config.js`; format with Prettier defaults (2-space indent, single quotes per config).

## Testing Guidelines
- No automated tests are defined yet; rely on `npm run lint` before commits.
- When adding tests, mirror page/component folders for colocated specs and name them `*.test.ts(x)`; use Vitest/React Testing Library if introduced.
- Add API contract checks for new routes (supertest) alongside route files when backend tests are added.

## Commit & Pull Request Guidelines
- Git history is sparse; use clear, present-tense messages. Conventional Commits (`feat:`, `fix:`, `chore:`) are encouraged for clarity.
- In PRs, include: brief summary of change, linked issue/task ID, screenshots or short screen captures for UI changes, and notes on env/setup impacts.
- Run `npm run lint` (and backend seed/start checks if touched) before requesting review; call out any follow-up work explicitly.

## Security & Configuration Tips
- Do not commit `.env` files. Required keys: frontend `.env` for client settings (if added), backend `.env` for `MONGODB_URI`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`.
- Keep sample configs minimal; prefer `.env.example` when sharing defaults.
- The API exposes `/health`; avoid enabling it on public hosts without auth or rate limits.
