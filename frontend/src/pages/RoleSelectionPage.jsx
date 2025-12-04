/**
 * Role Selection Page
 * Presents Admin vs Student entry points before authentication
 */

import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Shield, GraduationCap, ArrowRight } from 'lucide-react';

import { useAuth } from '../contexts/AuthContext';

const roleOptions = [
  {
    key: 'admin',
    title: 'Admin',
    description:
      'As an admin, you have the power to manage, monitor, and control every feature seamlessly.',
    icon: Shield,
    accent: 'border-[#f5c754] hover:bg-[#f5c754]/10 focus-visible:ring-[#f5c754]/40',
    actionLabel: 'Go to admin login',
  },
  {
    key: 'student',
    title: 'Student',
    description:
      'With a student account, you can discover new tasks, track progress, and unlock your potential every day.',
    icon: GraduationCap,
    accent: 'border-[#6eff8b] hover:bg-[#6eff8b]/10 focus-visible:ring-[#6eff8b]/40',
    actionLabel: 'Go to student portal',
  },
];

const RoleSelectionPage = () => {
  const navigate = useNavigate();
  const { isAuthenticated, getDashboardRoute } = useAuth();

  useEffect(() => {
    if (isAuthenticated) {
      navigate(getDashboardRoute(), { replace: true });
    }
  }, [isAuthenticated, getDashboardRoute, navigate]);

  const handleSelect = (target) => {
    if (target === 'admin') {
      navigate('/admin/login');
      return;
    }

    navigate('/school-portal');
  };

  return (
    <div className="min-h-screen bg-[#050608] text-white flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-xl space-y-10">
        <motion.div
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="space-y-3"
        >
          <p className="text-sm uppercase tracking-[0.35em] text-white/50">INAI</p>
          <h1 className="text-3xl font-semibold leading-tight">
            What&apos;s <br />
            <span className="text-white/80">features suits you ?</span>
          </h1>
          <p className="text-sm text-white/60">
            Pick the entry that matches your role to continue. Admins are guided through onboarding on their
            first login, while students head directly to the portal login.
          </p>
        </motion.div>

        <div className="space-y-6">
          {roleOptions.map(({ key, title, description, icon: Icon, accent, actionLabel }) => (
            <motion.button
              key={key}
              onClick={() => handleSelect(key)}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: key === 'student' ? 0.1 : 0 }}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              className={`w-full rounded-3xl border-2 bg-[#0c0f16] px-6 py-6 text-left transition focus:outline-none focus-visible:ring-2 ${accent}`}
            >
              <div className="flex items-center gap-4">
                <div className="rounded-2xl bg-white/5 p-4">
                  <Icon className="h-8 w-8" />
                </div>
                <div className="flex-1 space-y-1">
                  <p className="text-xs uppercase tracking-[0.2em] text-white/50">{actionLabel}</p>
                  <h2 className="text-2xl font-semibold">{title}</h2>
                  <p className="text-sm text-white/70">{description}</p>
                </div>
                <ArrowRight className="h-5 w-5 text-white/50" />
              </div>
            </motion.button>
          ))}
        </div>

        <p className="text-center text-xs text-white/40">
          Need help choosing? Contact your administrator or INAI support.
        </p>
      </div>
    </div>
  );
};

export default RoleSelectionPage;