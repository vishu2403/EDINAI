# Backend Documentation – INAI Education Management System

## Table of Contents
1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [Environment Configuration](#environment-configuration)
4. [Application Lifecycle](#application-lifecycle)
5. [Database Layer](#database-layer)
   - [ORM Models](#orm-models)
   - [Automatic Seeding](#automatic-seeding)
6. [Pydantic Schemas](#pydantic-schemas)
7. [Utility Modules](#utility-modules)
8. [API Routers](#api-routers)
   - [Authentication Router](#authentication-router)
   - [Admin Management Router](#admin-management-router)
   - [Member Management Router](#member-management-router)
   - [Onboarding Router](#onboarding-router)
   - [Dashboard Router](#dashboard-router)
9. [Security & Authorization](#security--authorization)
10. [Error Handling & Logging](#error-handling--logging)
11. [Running the Backend Locally](#running-the-backend-locally)
12. [Testing & Further Development](#testing--further-development)

---

## Overview
The backend is a FastAPI application that powers the INAI Education Management System. It provides RESTful endpoints for authentication, administrator/member management, onboarding workflows, and role-specific dashboards. Data persistence is handled through SQLAlchemy ORM backed by a SQLite database by default.

## Technology Stack
- **Framework**: FastAPI
- **Database ORM**: SQLAlchemy
- **Database**: SQLite (configurable)
- **Validation**: Pydantic
- **Authentication**: JSON Web Tokens (JWT) via `python-jose`
- **Password Hashing**: `passlib[bcrypt]`
- **Environment Loading**: `python-dotenv`

All backend dependencies are listed in `requirements.txt`.

## Environment Configuration
Configuration values are centralized in `app/config.py`. The module loads variables from the shell or a `.env` file using `python-dotenv` and exposes:

```env
SECRET_KEY=change-me-to-a-long-random-string
DATABASE_URL=sqlite:///./inai_db.sqlite
ACCESS_TOKEN_EXPIRE_DAYS=7
```

- **SECRET_KEY**: JWT signing key (must be strong and secret in production).
- **DATABASE_URL**: Any SQLAlchemy-supported URL (SQLite by default).
- **ACCESS_TOKEN_EXPIRE_DAYS**: Controls JWT expiration horizon.

## Application Lifecycle
`app/main.py` creates the FastAPI instance, configures CORS, mounts the uploads directory, and registers routers. A lifespan context manager executes on startup to create database tables, seed packages, and ensure a super-admin exists. All requests pass through a lightweight logging middleware for observability.

Key endpoints exposed by the base app:
- `GET /` – API welcome payload.
- `GET /health` – Health check for monitoring.
- Auto-generated docs: `/docs` (Swagger UI) and `/redoc`.

## Database Layer
### ORM Models
Defined in `app/models.py`:
- **Admin** – Platform administrators (super admin flag, package info, expiry tracking, login metadata).
- **Member** – Admin-managed staff or students with role IDs, work type enum, and status flags.
- **Package** – Subscription tiers constraining feature limits.
- **Contact** – Admin contact person details gathered during onboarding.
- **EducationCenter** – Institution profile including media references.
- **INAICredential** – Legacy credential storage (cleaned during onboarding completion).
- **RoleSequence** – Tracks sequential counters for work-type-based role IDs.

Each relationship uses cascading rules so deleting an admin removes dependent records.

### Automatic Seeding
`app/database.py` exposes `create_database_and_tables()`, invoked at startup to:
1. Create the uploads directory (if missing).
2. Create all tables via SQLAlchemy metadata.
3. Populate default packages (`p1`, `p2`, `p3`).
4. Create or refresh the default super admin (`admin@inai.edu`, `superadmin123`).

Foreign key enforcement for SQLite is enabled via an engine event hook.

## Pydantic Schemas
`app/schemas.py` defines request and response payloads:
- Auth-specific schemas (`LoginRequest`, `ChangePasswordRequest`, etc.).
- Admin/Member CRUD payloads (`AdminCreate`, `MemberUpdate`, etc.).
- Shared `ResponseBase` envelope wrapping API responses.
- Onboarding (`CompleteOnboardingRequest/Response`) and dashboard response schemas.

Validators enforce password length, package enumerations, and date formats.

## Utility Modules
- **`app/utils/universal_jwt_handler.py`**: Central JWT functionality—token creation, verification, and FastAPI dependencies that enforce admin/super-admin/member access as well as onboarding state guards.
- **`app/utils/role_generator.py`**: Generates unique role IDs per work type with concurrency-safe sequences.
- **`app/utils/file_handler.py`**: Currently placeholder helpers for future file upload persistence.

## API Routers
Routers reside in `app/routers/` and are mounted under their respective prefixes.

### Authentication Router
File: `auth.py`, prefix `/auth`
- `POST /login`: Authenticates admin or member, issues JWT, updates last-login timestamps, and enforces account status/expiry checks.
- `POST /change-password`: Validates the current password before updating credentials.
- `POST /forgot-password` & `POST /reset-password`: Demo-oriented reset flow generating and validating reset codes.
- `GET /me`: Returns current user context (role, profile fields) using JWT dependency injection.

### Admin Management Router
File: `admin_management.py`, prefix `/admin`
- CRUD operations for admins secured by `super_admin_required`.
- Subscription extension endpoint updates expiry dates and reactivates accounts when appropriate.
- CSV export endpoint returns a comma-separated list of admins including derived metrics like days until expiry.

### Member Management Router
File: `member_management.py`, prefix `/members`
- Admin-scoped CRUD operations filtered to the authenticated admin.
- Automatic role ID generation via `generate_role_id` utility.
- List endpoint supports pagination, filtering by work type, and active-only view.
- Toggle status and CSV export utilities for administrative tasks.

### Onboarding Router
File: `complete_onboarding.py`, prefix `/complete-onboarding`
- `POST /`: Consolidated onboarding flow that captures contact, education center, and INAI credential details in a single transaction, updates the admin’s login email/password, and issues a fresh JWT.
- `POST /upload-files`: Placeholder endpoint to accept multipart uploads for center media after initial onboarding.

### Dashboard Router
File: `dashboard.py`, prefix `/dashboard`
- Admin dashboard summarizing members, package usage, expiry, and activity metrics.
- Member dashboards for Chapter, Student, and Lecture roles with tailored insights (currently includes stubbed performance data where full metrics are pending).
- `GET /summary`: Lightweight counts and alerts (e.g., upcoming expiry) for quick UI badges.

## Security & Authorization
- JWT tokens signed with `SECRET_KEY` and default HS256 algorithm.
- Access dependencies (`admin_required`, `super_admin_required`, `member_required`) ensure role-based permissions.
- Additional onboarding guard dependencies ensure admins complete onboarding before accessing dashboards (or restrict onboarding endpoints once complete).
- Password hashing uses BCrypt via Passlib with safeguards for bcrypt’s maximum password length.
- Automatic deactivation occurs when an admin account is expired or inactive, propagating restrictions to members.

## Error Handling & Logging
- Global exception handler in `main.py` unifies unexpected errors into a consistent JSON structure.
- Per-endpoint error handling uses FastAPI’s `HTTPException` for predictable client responses.
- Request logging middleware prints method, path, status code, and latency to stdout—suitable for development and can be replaced with structured logging in production.

## Running the Backend Locally
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Create `.env`** (optional but recommended) with the secrets described earlier.
3. **Start the API server**
   ```bash
   uvicorn app.main:app --reload --port 8014
   ```
4. **Access documentation** at `http://localhost:8014/docs` or `http://localhost:8014/redoc`.

The first startup creates `inai_db.sqlite`, seeds packages, and ensures the super admin account exists.

## Testing & Further Development
- Add automated tests with `pytest`, `pytest-asyncio`, and `httpx` for API coverage.
- For schema/database changes, update the ORM models and recreate the SQLite database (development) or apply migrations if using a production-grade database (e.g., Alembic with PostgreSQL).
- Extend routers by defining new Pydantic schemas, ORM models, and endpoints following existing patterns.
- Consider elevating the placeholder file handling utilities into a full storage service (local filesystem, S3, etc.) before enabling uploads in production.

---

_This document reflects the backend state as of November 10, 2025. Update sections whenever routes, data models, or configuration behavior changes._
