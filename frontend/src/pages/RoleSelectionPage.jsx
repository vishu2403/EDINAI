import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ShieldCheck, GraduationCap } from 'lucide-react';

const ROLE_CARDS = [
  {
    key: 'admin',
    title: 'Admin',
    description:
      'As an Admin, you can manage, monitor, and control every feature seamlessly.',
    icon: ShieldCheck,
    gradient: 'from-yellow-500/40 via-amber-400/40 to-orange-400/40',
    border: 'border-yellow-400/70',
    text: 'text-yellow-300',
    onClickPath: '/admin/login',
  },
  {
    key: 'student',
    title: 'Student',
    description:
      'Explore new ideas, track progress, and unlock your potential every day.',
    icon: GraduationCap,
    gradient: 'from-emerald-500/40 via-teal-400/40 to-green-400/40',
    border: 'border-emerald-400/70',
    text: 'text-emerald-300',
    onClickPath: '/school-portal',
  },
];

const RoleSelectionPage = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#050507] flex items-center justify-center px-4 py-12">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative w-full max-w-3xl overflow-hidden rounded-[32px] border border-white/10 bg-white/[0.04] p-10 md:p-12 backdrop-blur-2xl shadow-[0_24px_96px_rgba(0,0,0,0.45)]"
      >
        <div className="absolute -top-24 -right-24 h-56 w-56 rounded-full bg-gradient-to-br from-white/20 to-transparent blur-3xl" />
        <div className="absolute -bottom-32 -left-24 h-72 w-72 rounded-full bg-gradient-to-br from-emerald-500/10 to-transparent blur-3xl" />

        <div className="relative text-center">
          <span className="text-xs font-semibold uppercase tracking-[0.4em] text-white/40">
            Choose Your Experience
          </span>
          <h1 className="mt-6 text-3xl font-semibold text-white md:text-4xl">
            What’s features suits you?
          </h1>
          <p className="mt-4 text-sm text-white/60 md:text-base">
            Pick the portal you want to explore. You can always switch later from the main menu.
          </p>
        </div>

        <div className="relative mt-12 space-y-6">
          {ROLE_CARDS.map((card) => {
            const Icon = card.icon;
            return (
              <motion.button
                key={card.key}
                type="button"
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                onClick={() => navigate(card.onClickPath)}
                className={`group relative w-full overflow-hidden rounded-3xl border ${card.border} bg-black/60 p-6 text-left transition-all duration-300 hover:border-opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/20 focus-visible:ring-offset-2 focus-visible:ring-offset-black`}
              >
                <div
                  className={`absolute inset-0 rounded-[inherit] bg-gradient-to-br ${card.gradient} opacity-0 transition-opacity duration-300 group-hover:opacity-100`}
                />
                <div className="relative flex flex-col gap-6 sm:flex-row sm:items-center">
                  <div className={`flex h-20 w-20 shrink-0 items-center justify-center rounded-2xl bg-black/40 border ${card.border}`}>
                    <Icon className={`h-10 w-10 ${card.text}`} />
                  </div>
                  <div className="flex-1">
                    <h2 className="text-2xl font-semibold text-white">{card.title}</h2>
                    <p className="mt-2 text-sm leading-relaxed text-white/70 md:text-base">
                      {card.description}
                    </p>
                    <span className={`mt-6 inline-flex items-center gap-2 text-sm font-medium transition-colors duration-300 group-hover:text-white ${card.text}`}>
                      Enter {card.title} Portal
                    </span>
                  </div>
                </div>
              </motion.button>
            );
          })}
        </div>
      </motion.div>
    </div>
  );
};

export default RoleSelectionPage;
