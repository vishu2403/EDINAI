import React, { useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import clsx from 'clsx';
import { Menu, X, LogOut } from 'lucide-react';

import Button from '../ui/Button';
import { useNewAdminAuth } from '../../contexts/NewAdminAuthContext.jsx';

const DEFAULT_LINKS = [
  { label: 'Home', path: '/admin/dashboard' },
  { label: 'All Members', path: '/admin/all-members' },
  { label: 'Terms & Condition', path: null },
  { label: 'Privacy Policy', path: null },
  { label: 'Theme', path: null },
  { label: 'Help Center', path: null },
];

const AdminShell = ({ title, description, children, links = DEFAULT_LINKS }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { admin, logout } = useNewAdminAuth();

  const [sidebarOpen, setSidebarOpen] = useState(false);

  const activePath = location.pathname;

  const sidebarLinks = useMemo(() => links, [links]);

  const handleNavigate = (item) => {
    if (item?.path) {
      navigate(item.path);
    }
    setSidebarOpen(false);
  };

  return (
    <div className="min-h-screen bg-[#050507] text-white">
      <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="medium"
              icon={sidebarOpen ? X : Menu}
              onClick={() => setSidebarOpen((prev) => !prev)}
            />
            <div>
              <h1 className="text-3xl font-semibold">{title}</h1>
              <p className="text-dark-text-muted">
                {description ||
                  (title === 'Dashboard'
                    ? 'Here’s a quick overview of your portal today.'
                    : 'Manage your team and portal access effortlessly.')}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full bg-primary-500 flex items-center justify-center text-lg font-semibold">
              {admin?.full_name?.charAt(0)?.toUpperCase() || 'A'}
            </div>
            <Button variant="secondary" size="medium" icon={LogOut} onClick={logout}>
              Logout
            </Button>
          </div>
        </header>

        <main>{children}</main>
      </div>

      {/* Sidebar overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            className="fixed inset-0 z-40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div
              className="absolute inset-0 bg-black/50"
              onClick={() => setSidebarOpen(false)}
            />
            <motion.aside
              className="absolute top-0 left-0 h-full w-72 bg-white text-gray-900 shadow-2xl flex flex-col"
              initial={{ x: -320 }}
              animate={{ x: 0 }}
              exit={{ x: -320 }}
              transition={{ type: 'spring', stiffness: 320, damping: 30 }}
            >
              <div className="px-8 pt-10 pb-6 border-b border-gray-200">
                <div className="text-3xl font-bold tracking-wide">INAI</div>
                <div className="text-sm text-gray-500 tracking-[0.5em] uppercase mt-1">
                  Verse
                </div>
              </div>

              <nav className="flex-1 overflow-y-auto px-8 py-6 space-y-1">
                {sidebarLinks.map((link) => {
                  const isActive = link.path && activePath === link.path;
                  const isDisabled = !link.path;

                  return (
                    <button
                      key={link.label}
                      type="button"
                      onClick={() => handleNavigate(link)}
                      disabled={isDisabled}
                      className={clsx(
                        'w-full text-left px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                        isActive
                          ? 'bg-primary-600 text-white shadow'
                          : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100',
                        isDisabled && 'cursor-not-allowed opacity-50'
                      )}
                    >
                      {link.label}
                    </button>
                  );
                })}
              </nav>

              <div className="px-8 py-6 border-t border-gray-200">
                <Button
                  variant="ghost"
                  size="medium"
                  icon={LogOut}
                  className="text-gray-700 hover:text-gray-900"
                  onClick={() => {
                    logout();
                    setSidebarOpen(false);
                  }}
                >
                  Logout
                </Button>
              </div>
            </motion.aside>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AdminShell;
