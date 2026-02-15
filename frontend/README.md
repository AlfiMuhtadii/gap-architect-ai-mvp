# Frontend Setup

## Prerequisites
- Node.js 20+
- pnpm

## 1. Environment
- Copy template:
  - `cp .env.dev.example .env`
- Set API base:
  - `NEXT_PUBLIC_API_BASE=http://localhost:8000`

## 2. Install Dependencies
```bash
pnpm install
```

## 3. Run App
```bash
pnpm dev
```

## 4. Verify
- Frontend: `http://localhost:3000`
- Backend API expected at: `http://localhost:8000`

## Type Check
```bash
pnpm exec tsc --noEmit
```
