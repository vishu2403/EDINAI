import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { MathJaxContext } from 'better-react-mathjax'
import App from './App.jsx'
import { AuthProvider } from './contexts/AuthContext.jsx'
import { NewAdminAuthProvider } from './contexts/NewAdminAuthContext.jsx'
import './index.css'

const mathJaxConfig = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
  },
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <MathJaxContext version={3} config={mathJaxConfig}>
      <BrowserRouter>
        <AuthProvider>
          <NewAdminAuthProvider>
            <App />
            <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#1a1f2e',
                color: '#ffffff',
                border: '1px solid #2d3748',
                borderRadius: '12px',
                fontSize: '14px',
              },
              success: {
                iconTheme: {
                  primary: '#48bb78',
                  secondary: '#ffffff',
                },
              },
              error: {
                iconTheme: {
                  primary: '#f56565',
                  secondary: '#ffffff',
                },
              },
            }}
          />
          </NewAdminAuthProvider>
        </AuthProvider>
      </BrowserRouter>
    </MathJaxContext>
  </React.StrictMode>,
)
