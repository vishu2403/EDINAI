# INAI Education Management System - Complete Documentation

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Features](#features)
4. [Technology Stack](#technology-stack)
5. [Installation & Setup](#installation--setup)
6. [API Documentation](#api-documentation)
7. [Database Schema](#database-schema)
8. [Frontend Documentation](#frontend-documentation)
9. [Authentication & Security](#authentication--security)
10. [File Structure](#file-structure)
11. [Development Guide](#development-guide)
12. [Production Deployment](#production-deployment)
13. [Troubleshooting](#troubleshooting)

## 🎯 Project Overview

The **INAI Education Management System** is a comprehensive web application designed to manage educational institutions, their administrators, members, and educational content. The system provides role-based access control, onboarding flows, and dashboard analytics.

### Key Components
- **Backend**: FastAPI-based REST API with SQLite database
- **Frontend**: React SPA with modern UI/UX
- **Authentication**: JWT-based with role hierarchy
- **File Management**: Secure upload and storage system
- **Onboarding**: Multi-step wizard for new administrators

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React SPA     │    │   FastAPI API   │    │  SQLite Database│
│   (Frontend)    │◄──►│   (Backend)     │◄──►│   (Storage)     │
│  Port 3000      │    │  Port 8014      │    │  Local File     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
    ┌────▼────┐             ┌────▼────┐             ┌────▼────┐
    │Tailwind │             │ SQLAlch │             │ Package │
    │Framer   │             │ Pydantic│             │ Uploads │
    │React    │             │ JWT     │             │ Files   │
    │Router   │             │ Bcrypt  │             │         │
    └─────────┘             └─────────┘             └─────────┘
```

### Communication Flow
1. **User** interacts with React frontend
2. **Frontend** makes HTTP requests to FastAPI backend
3. **Backend** processes requests, validates JWT tokens
4. **Database** stores/retrieves data via SQLAlchemy ORM
5. **Files** uploaded to secure directory structure

## ✨ Features

### 🔐 Authentication & Authorization
- **JWT-based authentication** with secure token management
- **Role hierarchy**: Super Admin → Admin → Member
- **Multi-role support**: Admin, Chapter, Student, Lecture members
- **Account expiry management** with automatic deactivation
- **Password security** with bcrypt hashing

### 👥 User Management
- **Admin Management**: Super admins can create/manage other admins
- **Member Management**: Admins can create/manage their team members
- **Role-based access control** throughout the system
- **Account status management** (active/inactive)

### 📚 Onboarding System
New single-step onboarding workflow for administrators that captures all required data in one screen and one API call.

#### Frontend Flow
- Route: `/complete-onboarding`
- Component: `CompleteOnboardingPage.jsx`
- Inputs collected:
  - **Contact Person**: first name, last name, phone, designation, address, date of birth (email comes from the authenticated admin account automatically)
  - **Education Center**: center name, hero/cover image, gallery photos, logo, activity images
  - **INAI Credentials**: INAI email + password (becomes the admin's login email/password)
- After successful submission the user receives a fresh JWT, the context is refreshed, and the admin is redirected to `/admin-dashboard`.

#### Backend Endpoint
- Method & Path: `POST /api/complete-onboarding/`
- Request body (JSON):
  ```json
  {
    "first_name": "Asha",
    "last_name": "Patel",
    "address": "55, Dhruv Park, Surat",
    "designation": "Teacher",
    "phone_number": "+91-9328883977",
    "date_of_birth": "05-03-1995",
    "education_center_name": "INAI Worlds",
    "upload_image": "data:image/png;base64,...",
    "center_photos": ["data:image/png;base64,..."],
    "logo": "data:image/png;base64,...",
    "other_activities": ["data:image/png;base64,..."],
    "inai_email": "inai.portal@example.com",
    "inai_password": "SecurePass@123"
  }
  ```
- Response payload:
  ```json
  {
    "status": true,
    "message": "Onboarding completed successfully!",
    "data": {
      "contact": { /* saved contact details */ },
      "education_center": { /* saved center details */ },
      "onboarding_completed": true,
      "new_token": "<JWT>",
      "admin_info": {
        "admin_id": 12,
        "name": "Asha Patel",
        "email": "inai.portal@example.com",
        "package": "p2",
        "has_inai_credentials": true
      }
    }
  }
  ```
- Optional file uploads remain available via `POST /api/complete-onboarding/upload-files` for center media after onboarding.

During onboarding the admin's account email/password are replaced with the INAI credentials provided, so the admin can immediately use them to access the platform. Any legacy entries in `inai_credentials` are removed. Password changes can still be performed later via `/auth/change-password` if required.

### 📊 Dashboard Analytics
- **Admin Dashboard**: System overview, member statistics, management tools
- **Member Dashboards**: Role-specific views for Chapter/Student/Lecture
- **Real-time metrics**: User counts, progress tracking, performance data
- **Package limits**: Display current package restrictions

### 📁 File Management
- **Secure file uploads** with validation
- **Multiple file types** supported (images, documents)
- **File size restrictions** based on package limits
- **Organized storage** structure with cleanup on errors

### 🎨 Modern UI/UX
- **Dark neumorphic theme** matching provided designs
- **Responsive design** for all device sizes
- **Smooth animations** with Framer Motion
- **Intuitive navigation** with role-based menus

## 💻 Technology Stack

### Backend
- **FastAPI** 0.104.1 - Modern Python web framework
- **SQLAlchemy** 2.0.23 - Python ORM
- **SQLite** - Lightweight database
- **Pydantic** 2.5.0 - Data validation
- **Python-Jose** 3.3.0 - JWT handling
- **Bcrypt** 4.0.1 - Password hashing
- **Python-multipart** 0.0.6 - File uploads
- **Uvicorn** 0.24.0 - ASGI server

### Frontend
- **React** 18.2.0 - UI library
- **Vite** 4.5.14 - Build tool
- **React Router** 6.8.0 - Client-side routing
- **Tailwind CSS** 3.4.0 - Utility-first CSS
- **Framer Motion** 10.12.0 - Animation library
- **Axios** 1.3.0 - HTTP client
- **React Hook Form** 7.43.0 - Form management
- **React Hot Toast** 2.4.0 - Notifications
- **Lucide React** 0.263.0 - Icons

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.8+**
- **Node.js 16+**
- **npm** or **yarn**

### Backend Setup

1. **Clone Repository**
   ```bash
   cd d:\kamla\auth_f
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create `.env` file (optional):
   ```env
   SECRET_KEY=your-secret-key-here
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   ```

5. **Start Backend Server**
   ```bash
   uvicorn app.main:app --reload --port 8014
   ```

### Frontend Setup

1. **Navigate to Frontend**
   ```bash
   cd frontend
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Start Development Server**
   ```bash
   npm run dev
   ```

### Default Login Credentials
- **Email**: admin@inai.edu
- **Password**: superadmin123
- **Role**: Super Administrator

## 🔌 API Documentation

### Base URL
```
http://localhost:8014
```

### Authentication Endpoints

#### POST /auth/login
Login for admins and members
```json
{
  "email": "admin@inai.edu",
  "password": "superadmin123"
}
```

#### POST /auth/change-password
Change password (requires authentication)
```json
{
  "old_password": "current_password",
  "new_password": "new_password"
}
```

#### GET /auth/me
Get current user information (requires authentication)

### Admin Management Endpoints

#### GET /admin-management/
Get all admins (super admin only)

#### POST /admin-management/
Create new admin (super admin only)
```json
{
  "name": "Admin Name",
  "email": "admin@example.com",
  "package": "p1",
  "expiry_months": 12
}
```

#### PUT /admin-management/{admin_id}
Update admin details (super admin only)

#### DELETE /admin-management/{admin_id}
Delete admin (super admin only)

### Member Management Endpoints

#### GET /member-management/
Get all members for current admin

#### POST /member-management/
Create new member
```json
{
  "name": "Member Name",
  "email": "member@example.com",
  "work_type": "chapter"
}
```

#### PUT /member-management/{member_id}
Update member details

#### DELETE /member-management/{member_id}
Delete member

#### PUT /member-management/{member_id}/toggle-status
Toggle member active status

### Onboarding Endpoints

#### GET /contact/
Get contact information

#### POST /contact/
Save contact information
```json
{
  "full_name": "John Doe",
  "phone": "+1234567890",
  "address": "123 Main St",
  "city": "City",
  "state": "State",
  "country": "Country",
  "postal_code": "12345"
}
```

#### GET /education-center/
Get education center information

#### POST /education-center/
Create education center with file uploads (multipart/form-data)

#### GET /inai-credentials/
Get INAI credentials

#### POST /inai-credentials/
Complete onboarding with INAI credentials
```json
{
  "username": "inai_username",
  "password": "inai_password"
}
```

### Dashboard Endpoints

#### GET /dashboard/admin-dashboard
Get admin dashboard metrics

#### GET /dashboard/chapter-dashboard
Get chapter dashboard metrics

#### GET /dashboard/student-dashboard
Get student dashboard metrics

#### GET /dashboard/lecture-dashboard
Get lecture dashboard metrics

## 🗄️ Database Schema

### Tables Overview

#### Admin Table
```sql
CREATE TABLE admin (
    admin_id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    package VARCHAR(10) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    expiry_date TIMESTAMP NOT NULL,
    has_inai_credentials BOOLEAN DEFAULT FALSE,
    active BOOLEAN DEFAULT TRUE,
    is_super_admin BOOLEAN DEFAULT FALSE,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Member Table
```sql
CREATE TABLE member (
    member_id INTEGER PRIMARY KEY,
    admin_id INTEGER NOT NULL,
    role_id VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    work_type VARCHAR(20) NOT NULL, -- 'chapter', 'student', 'lecture'
    active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admin_id) REFERENCES admin (admin_id)
);
```

#### Package Table
```sql
CREATE TABLE package (
    package_id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    max_members INTEGER NOT NULL,
    max_file_size_mb INTEGER NOT NULL,
    max_files_per_upload INTEGER NOT NULL,
    max_minutes_per_lecture INTEGER,
    ai_videos_per_lecture INTEGER,
    topics_per_lecture INTEGER,
    max_quality VARCHAR(10)
);
```

#### Contact Table
```sql
CREATE TABLE contact (
    contact_id INTEGER PRIMARY KEY,
    admin_id INTEGER UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    phone VARCHAR(20) NOT NULL,
    address TEXT NOT NULL,
    city VARCHAR(50) NOT NULL,
    state VARCHAR(50) NOT NULL,
    country VARCHAR(50) NOT NULL,
    postal_code VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admin_id) REFERENCES admin (admin_id)
);
```

#### Education Center Table
```sql
CREATE TABLE education_center (
    center_id INTEGER PRIMARY KEY,
    admin_id INTEGER UNIQUE NOT NULL,
    center_name VARCHAR(100) NOT NULL,
    year_established INTEGER NOT NULL,
    student_count INTEGER NOT NULL,
    teacher_count INTEGER NOT NULL,
    upload_image VARCHAR(255),
    center_photos TEXT, -- JSON array of file paths
    logo VARCHAR(255),
    other_activities TEXT, -- JSON array of file paths
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admin_id) REFERENCES admin (admin_id)
);
```

#### INAI Credential Table
```sql
CREATE TABLE inai_credential (
    credential_id INTEGER PRIMARY KEY,
    admin_id INTEGER UNIQUE NOT NULL,
    username VARCHAR(50) NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admin_id) REFERENCES admin (admin_id)
);
```

### Pre-populated Data

#### Package Types
- **p1**: Basic (5 members, 10MB files, 2 files/upload)
- **p2**: Standard (15 members, 50MB files, 5 files/upload, 30min lectures)
- **p3**: Premium (50 members, 100MB files, 10 files/upload, 60min lectures)

## 🎨 Frontend Documentation

### Project Structure
```
frontend/
├── public/                 # Static assets
├── src/
│   ├── components/         # Reusable components
│   │   └── ui/            # Base UI components
│   │       ├── Button.jsx
│   │       ├── InputField.jsx
│   │       ├── FileUploader.jsx
│   │       └── LoadingSpinner.jsx
│   ├── contexts/          # React contexts
│   │   └── AuthContext.jsx
│   ├── pages/             # Page components
│   │   ├── auth/          # Authentication pages
│   │   │   └── LoginPage.jsx
│   │   ├── onboarding/    # Onboarding flow
│   │   │   ├── ChangePasswordPage.jsx
│   │   │   ├── ContactPersonPage.jsx
│   │   │   ├── EducationCenterPage.jsx
│   │   │   └── INAICredentialsPage.jsx
│   │   ├── admin/         # Admin pages
│   │   │   ├── AdminDashboard.jsx
│   │   │   ├── AddMemberPage.jsx
│   │   │   └── AllMembersPage.jsx
│   │   ├── member/        # Member dashboards
│   │   │   ├── ChapterDashboard.jsx
│   │   │   ├── StudentDashboard.jsx
│   │   │   └── LectureDashboard.jsx
│   │   └── NotFoundPage.jsx
│   ├── utils/             # Utility functions
│   │   └── api.js         # API client
│   ├── App.jsx            # Main app component
│   ├── main.jsx           # Entry point
│   └── index.css          # Global styles
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

### Key Components

#### AuthContext
Manages authentication state globally:
```jsx
const { user, login, logout, isAuthenticated, isLoading } = useAuth();
```

#### API Client (api.js)
Centralized HTTP client with interceptors:
```javascript
// Automatically adds Authorization headers
// Handles token refresh and errors
import { authAPI, adminAPI, memberAPI } from './utils/api';
```

#### Protected Routing
Routes are protected based on authentication and roles:
```jsx
<ProtectedRoute requireAuth={true} allowedRoles={['admin']}>
  <AdminDashboard />
</ProtectedRoute>
```

### Styling System

#### Tailwind Configuration
Custom dark theme with neumorphic shadows:
```javascript
// tailwind.config.js
theme: {
  extend: {
    colors: {
      'dark-bg': '#0a0f1a',
      'dark-card': '#1a1f2e',
      'dark-border': '#2d3748'
    },
    boxShadow: {
      'neumorphic': '8px 8px 16px rgba(0, 0, 0, 0.3), -8px -8px 16px rgba(255, 255, 255, 0.05)'
    }
  }
}
```

#### Custom CSS Classes
```css
.card {
  @apply bg-dark-card border border-dark-border shadow-neumorphic;
}

.text-gradient {
  @apply bg-gradient-to-r from-blue-500 to-cyan-500 bg-clip-text text-transparent;
}
```

### State Management

#### Authentication State
- User information (name, email, role, permissions)
- JWT token management
- Onboarding completion status
- Role-based navigation

#### Form Management
- React Hook Form for validation
- Real-time error display
- Loading states
- Success/error notifications

## 🔒 Authentication & Security

### JWT Token Structure
```json
{
  "role": "admin|member",
  "id": 1,
  "package": "p1|p2|p3",
  "has_inai_credentials": true,
  "is_super_admin": false,
  "exp": 1699372800
}
```

### Security Features
- **Password hashing** with bcrypt (4.0.1)
- **JWT tokens** with configurable expiration
- **CORS protection** for cross-origin requests
- **File upload validation** (type, size, security)
- **SQL injection prevention** via SQLAlchemy ORM
- **XSS protection** through proper input sanitization

### Role-Based Access Control

#### Permission Matrix
| Feature | Super Admin | Admin | Chapter | Student | Lecture |
|---------|-------------|-------|---------|---------|---------|
| Create Admins | ✅ | ❌ | ❌ | ❌ | ❌ |
| Manage Members | ✅ | ✅ | ❌ | ❌ | ❌ |
| View Dashboard | ✅ | ✅ | ✅ | ✅ | ✅ |
| Upload Files | ✅ | ✅ | ❌ | ❌ | ❌ |
| Access Analytics | ✅ | ✅ | ✅ | ❌ | ✅ |

### Password Security
- **Minimum requirements** enforced in frontend
- **Bcrypt hashing** with salt rounds
- **Password strength indicator**
- **Secure password reset** flow

## 📁 File Structure

```
auth_f/
├── app/                           # Backend application
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── database.py               # Database setup and initialization
│   ├── models.py                 # SQLAlchemy models
│   ├── schemas.py                # Pydantic models
│   ├── routers/                  # API route handlers
│   │   ├── __init__.py
│   │   ├── auth.py              # Authentication routes
│   │   ├── admin_management.py   # Admin CRUD routes
│   │   ├── member_management.py  # Member CRUD routes
│   │   ├── contact.py           # Contact info routes
│   │   ├── education_center.py  # Education center routes
│   │   ├── inai_credentials.py  # INAI credentials routes
│   │   └── dashboard.py         # Dashboard routes
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── universal_jwt_handler.py  # JWT utilities
│       ├── role_generator.py         # Role ID generation
│       └── file_handler.py           # File upload utilities
├── frontend/                    # React frontend application
│   ├── public/                  # Static assets
│   ├── src/                     # Source code (detailed above)
│   ├── package.json            # Node dependencies
│   ├── vite.config.js          # Vite configuration
│   ├── tailwind.config.js      # Tailwind configuration
│   └── postcss.config.js       # PostCSS configuration
├── uploads/                     # File upload storage
│   ├── images/                  # Image uploads
│   ├── photos/                  # Photo uploads
│   ├── logos/                   # Logo uploads
│   └── activities/              # Activity uploads
├── data.db                      # SQLite database file
├── requirements.txt             # Python dependencies
├── README.md                    # Project README
└── DOCUMENTATION.md             # This documentation
```

## 🛠️ Development Guide

### Setting Up Development Environment

1. **Backend Development**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run with auto-reload
   uvicorn app.main:app --reload --port 8014
   
   # View API documentation
   # http://localhost:8014/docs
   ```

2. **Frontend Development**
   ```bash
   cd frontend
   npm install
   npm run dev
   
   # Frontend available at http://localhost:3000
   ```

### Adding New Features

#### Backend - New API Endpoint
1. **Create route in appropriate router**
   ```python
   @router.post("/new-endpoint")
   async def new_endpoint(data: NewSchema, db: Session = Depends(get_db)):
       # Implementation
       pass
   ```

2. **Add Pydantic schema**
   ```python
   class NewSchema(BaseModel):
       field1: str
       field2: int
   ```

3. **Update database model if needed**
   ```python
   class NewTable(Base):
       __tablename__ = "new_table"
       id = Column(Integer, primary_key=True)
   ```

#### Frontend - New Page
1. **Create page component**
   ```jsx
   // src/pages/NewPage.jsx
   const NewPage = () => {
     return <div>New Page Content</div>;
   };
   ```

2. **Add route to App.jsx**
   ```jsx
   <Route path="/new-page" element={<NewPage />} />
   ```

3. **Add API function**
   ```javascript
   // src/utils/api.js
   export const newAPI = {
     getData: () => apiClient.get('/new-endpoint')
   };
   ```

### Database Migrations

For schema changes:
1. **Backup current database**
   ```bash
   cp data.db data.db.backup
   ```

2. **Update models.py**
3. **Delete database file** (for development)
4. **Restart server** to recreate tables

### Testing

#### Backend Testing
```bash
# Install testing dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

#### Frontend Testing
```bash
# Install testing dependencies
npm install --save-dev @testing-library/react @testing-library/jest-dom

# Run tests
npm test
```

## 🚀 Production Deployment

### Backend Deployment

#### Using Docker
1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Build and run**
   ```bash
   docker build -t inai-backend .
   docker run -p 8000:8000 inai-backend
   ```

#### Using Virtual Server
1. **Install dependencies**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. **Setup application**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure systemd service**
   ```ini
   [Unit]
   Description=INAI Backend
   After=network.target

   [Service]
   User=www-data
   WorkingDirectory=/path/to/app
   Environment="PATH=/path/to/app/venv/bin"
   ExecStart=/path/to/app/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

### Frontend Deployment

#### Build for Production
```bash
cd frontend
npm run build
```

#### Deploy to Static Hosting
1. **Netlify**
   ```bash
   # Connect GitHub repo
   # Set build command: npm run build
   # Set publish directory: dist
   ```

2. **Vercel**
   ```bash
   npm install -g vercel
   vercel --prod
   ```

3. **Apache/Nginx**
   ```bash
   # Copy dist/ contents to web server directory
   cp -r dist/* /var/www/html/
   ```

### Environment Variables

#### Production Backend
```env
SECRET_KEY=your-super-secure-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
CORS_ORIGINS=["https://yourdomain.com"]
DATABASE_URL=sqlite:///./production.db
```

#### Production Frontend
```env
VITE_API_URL=https://api.yourdomain.com
VITE_ENVIRONMENT=production
```

### Database Considerations

#### For Production
1. **Use PostgreSQL** instead of SQLite
2. **Setup regular backups**
3. **Configure connection pooling**
4. **Enable database logging**

```python
# For PostgreSQL
DATABASE_URL = "postgresql://user:password@localhost/dbname"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
```

### SSL/HTTPS Setup

#### Backend
```python
# Use HTTPS in production
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem"
    )
```

#### Frontend
Configure web server (Nginx/Apache) for HTTPS

### Performance Optimization

#### Backend
- Use connection pooling
- Implement caching (Redis)
- Optimize database queries
- Use CDN for file uploads

#### Frontend
- Enable compression (gzip)
- Use CDN for static assets
- Implement service workers
- Optimize bundle size

## 🐛 Troubleshooting

### Common Issues

#### Backend Issues

**Issue**: Server won't start
```bash
# Check Python version
python --version  # Should be 3.8+

# Check dependencies
pip list

# Check for port conflicts
netstat -an | findstr :8014
```

**Issue**: Database errors
```bash
# Delete and recreate database
rm data.db
# Restart server to recreate tables
```

**Issue**: File upload errors
```bash
# Check directory permissions
mkdir -p uploads/images uploads/photos uploads/logos uploads/activities
chmod 755 uploads/
```

#### Frontend Issues

**Issue**: Build fails
```bash
# Clear node modules
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 16+
```

**Issue**: API connection errors
```bash
# Check backend is running
curl http://localhost:8014/

# Check proxy configuration in vite.config.js
```

**Issue**: Styling problems
```bash
# Rebuild Tailwind
npm run build

# Check PostCSS configuration
```

#### Authentication Issues

**Issue**: Login fails
1. Check backend logs for errors
2. Verify password hash in database
3. Check JWT secret key configuration
4. Verify CORS settings

**Issue**: Token expires quickly
```python
# Increase token expiration in backend
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours
```

### Debug Mode

#### Backend Debugging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints
print(f"User data: {user_data}")
```

#### Frontend Debugging
```javascript
// Enable React DevTools
// Add console logs
console.log('API Response:', response.data);

// Check network tab for API calls
```

### Performance Issues

#### Slow Database Queries
```python
# Add database indexes
# Monitor query performance
# Use query optimization
```

#### Slow Frontend Loading
```javascript
// Check bundle size
npm run build -- --analyze

// Implement lazy loading
const LazyComponent = React.lazy(() => import('./Component'));
```

### Monitoring & Logging

#### Backend Monitoring
```python
# Add request logging middleware
# Monitor error rates
# Setup alerts for failures
```

#### Frontend Monitoring
```javascript
// Add error boundary
// Monitor user interactions
// Track performance metrics
```

## 📞 Support & Maintenance

### Regular Maintenance Tasks
- **Weekly**: Check server logs and performance
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Database maintenance and optimization
- **Annually**: Security audit and backup testing

### Backup Strategy
1. **Database backups**: Daily automated backups
2. **File uploads**: Weekly backup to cloud storage
3. **Configuration**: Version control all config files
4. **Documentation**: Keep documentation updated

### Update Process
1. **Test updates** in development environment
2. **Backup production** data
3. **Deploy updates** during maintenance window
4. **Verify functionality** post-deployment
5. **Monitor** for issues

---

## 📝 Conclusion

The INAI Education Management System is a comprehensive solution for educational institution management. This documentation provides all necessary information for development, deployment, and maintenance of the system.

**For additional support or questions, please refer to:**
- API Documentation: http://localhost:8014/docs
- Source Code: Review inline comments
- Issue Tracking: Use your version control system

**Project Version**: 1.0.0  
**Last Updated**: November 7, 2025  
**Maintained By**: Development Team

---

*This documentation is a living document and should be updated as the system evolves.*
