# INAI Education Management System - Frontend

A modern React frontend for the INAI Education Management System with dark theme, responsive design, and complete onboarding flow.

## 🚀 Features

- **Dark Theme UI** - Neumorphic design matching provided screenshots
- **Authentication System** - JWT-based auth with role management
- **Protected Routing** - Role-based access control
- **Onboarding Flow** - Multi-step wizard for new admins
- **File Upload** - Drag & drop with preview and validation
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Real-time Updates** - Dynamic dashboard data
- **Smooth Animations** - Framer Motion transitions

## 📋 Tech Stack

- **React 18** - Modern React with hooks
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Animation library
- **React Router** - Client-side routing
- **React Hook Form** - Form validation
- **Axios** - HTTP client
- **Lucide React** - Modern icon library

## 🛠️ Setup & Installation

1. **Install Dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```

3. **Build for Production**
   ```bash
   npm run build
   ```

4. **Preview Production Build**
   ```bash
   npm run preview
   ```

## 🏗️ Project Structure

```
frontend/
├── public/                 # Static assets
├── src/
│   ├── components/         # Reusable UI components
│   │   └── ui/            # Base UI components
│   ├── contexts/          # React contexts
│   ├── pages/             # Page components
│   │   ├── auth/          # Authentication pages
│   │   ├── onboarding/    # Onboarding flow
│   │   ├── admin/         # Admin pages
│   │   └── member/        # Member dashboards
│   ├── utils/             # Utility functions
│   ├── App.jsx            # Main app component
│   ├── main.jsx           # App entry point
│   └── index.css          # Global styles
├── package.json
├── vite.config.js
└── tailwind.config.js
```

## 🎨 Design System

### Colors
- **Primary**: Blue gradient (#3B82F6 to #06B6D4)
- **Dark Background**: #0a0f1a
- **Card Background**: #1a1f2e
- **Border**: #2d3748
- **Text Primary**: #ffffff
- **Text Secondary**: #a0aec0

### Components
- **Button** - Various styles (primary, secondary, danger, ghost)
- **InputField** - Form inputs with validation
- **FileUploader** - Drag & drop file upload
- **LoadingSpinner** - Animated loading states
- **Card** - Container with neumorphic shadow

## 🔐 Authentication Flow

1. **Login** - Unified login for admins and members
2. **Role Detection** - Automatic routing based on user role
3. **Onboarding** - Multi-step process for new admins:
   - Change Password
   - Contact Person Details
   - Education Center (with file uploads)
   - INAI Credentials (completion)
4. **Dashboard Access** - Role-specific dashboards

## 📱 Responsive Design

- **Mobile First** - Designed for mobile, enhanced for desktop
- **Breakpoints**:
  - sm: 640px
  - md: 768px
  - lg: 1024px
  - xl: 1280px

## 🛣️ Routing

### Public Routes
- `/login` - Login page

### Admin Routes (Protected)
- `/admin-dashboard` - Main admin dashboard
- `/add-member` - Add new team member
- `/all-members` - Manage all members

### Member Routes (Role-specific)
- `/chapter-dashboard` - Chapter management dashboard
- `/student-dashboard` - Student dashboard
- `/lecture-dashboard` - Lecture management dashboard

### Onboarding Routes (Admin only, pre-completion)
- `/change-password` - Password change
- `/contact-person` - Contact details
- `/education-center` - Center info & files
- `/inai-credentials` - Final credentials

## 🎯 Key Features

### Authentication
- JWT token management
- Automatic token refresh
- Role-based access control
- Protected route guards

### File Upload
- Drag & drop interface
- File type validation
- Size restrictions
- Preview functionality
- Progress indicators

### Forms
- Real-time validation
- Error handling
- Loading states
- Auto-save capabilities

### Animations
- Page transitions
- Loading animations
- Interactive elements
- Smooth micro-interactions

## 🔧 Configuration

### Environment Variables
Create `.env` file in frontend directory:
```env
VITE_API_URL=http://localhost:8000
```

### API Proxy
Vite is configured to proxy `/api` requests to the backend server.

## 🚀 Development

### Running with Backend
1. Start backend server: `uvicorn app.main:app --reload`
2. Start frontend: `npm run dev`
3. Access at: http://localhost:3000

### Available Scripts
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## 📦 Build & Deployment

### Production Build
```bash
npm run build
```

Output will be in `dist/` directory.

### Deployment Options
- **Netlify** - Connect GitHub repo for auto-deploy
- **Vercel** - Import project for instant deployment
- **Static Hosting** - Upload `dist/` folder to any static host

## 🎨 Customization

### Theme Colors
Edit `tailwind.config.js` to customize colors:
```js
theme: {
  extend: {
    colors: {
      primary: { /* your colors */ },
      dark: { /* your dark theme */ }
    }
  }
}
```

### Components
All UI components are in `src/components/ui/` and can be customized.

## 🐛 Troubleshooting

### Common Issues

1. **CORS Errors**
   - Ensure backend CORS is configured correctly
   - Check API URL in proxy configuration

2. **Build Errors**
   - Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
   - Check for TypeScript errors in JSX files

3. **Authentication Issues**
   - Clear localStorage: `localStorage.clear()`
   - Check JWT token format and expiry

### Browser Support
- Chrome 88+
- Firefox 85+
- Safari 14+
- Edge 88+

## 📝 Notes

- **CSS Lint Warnings**: The @tailwind and @apply directives are Tailwind-specific and processed by PostCSS. These warnings are expected and don't affect functionality.
- **File Uploads**: Ensure backend upload directory exists and has proper permissions
- **Security**: Never expose API keys or sensitive data in frontend code

## 🤝 Contributing

1. Follow React best practices
2. Use TypeScript for new components
3. Maintain consistent styling with Tailwind
4. Add proper error handling
5. Test on multiple screen sizes

Built with ❤️ using React, Tailwind CSS, and modern web technologies.
