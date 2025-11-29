/**
 * Lecture Dashboard - Dashboard for Lecture Management members
 */

import React, { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { dashboardAPI, lectureAPI } from '../../utils/api';
import {
  Bell,
  BookOpen,
  LayoutDashboard,
  LogOut,
  MessageSquare,
  Moon,
  PlayCircle,
  PlusCircle,
  Share2,
} from 'lucide-react';
import toast from 'react-hot-toast';

const LectureDashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lecturesLoading, setLecturesLoading] = useState(false);
  const [lectures, setLectures] = useState([]);
  const [lectureFilters, setLectureFilters] = useState({ std: '', subject: '' });
  const [selectedNav, setSelectedNav] = useState('Dashboard');

  useEffect(() => {
    loadDashboardData();
  }, []);

  useEffect(() => {
    if (selectedNav === 'All Lectures') {
      loadLectures(lectureFilters);
    }
  }, [selectedNav]);

  const loadDashboardData = async () => {
    try {
      const response = await dashboardAPI.getLectureDashboard();
      if (response.status) {
        setDashboardData(response.data);
      }
    } catch (error) {
      console.error('Dashboard data error:', error);
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const loadLectures = async (filters) => {
    try {
      setLecturesLoading(true);
      const response = await lectureAPI.listGeneratedLectures(filters);
      if (response.status) {
        setLectures(response.data?.items || []);
      }
    } catch (error) {
      console.error('Lecture list error:', error);
      toast.error(error.response?.data?.detail || 'Failed to load lectures');
    } finally {
      setLecturesLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/admin/login');
  };

  const handleNavClick = (item) => {
    setSelectedNav(item.label);
  };

  const filteredLectures = useMemo(() => {
    if (!lectureFilters.std && !lectureFilters.subject) {
      return lectures;
    }
    return lectures.filter((lecture) => {
      const matchesStd = lectureFilters.std
        ? lecture.std?.toLowerCase().includes(lectureFilters.std.toLowerCase())
        : true;
      const matchesSubject = lectureFilters.subject
        ? lecture.subject?.toLowerCase().includes(lectureFilters.subject.toLowerCase())
        : true;
      return matchesStd && matchesSubject;
    });
  }, [lectures, lectureFilters]);

  const handleFilterChange = (field, value) => {
    setLectureFilters((prev) => ({ ...prev, [field]: value }));
  };

  const handleApplyFilters = () => {
    loadLectures(lectureFilters);
  };

  const handleClearFilters = () => {
    const cleared = { std: '', subject: '' };
    setLectureFilters(cleared);
    loadLectures(cleared);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#050505] flex items-center justify-center">
        <div className="spinner w-8 h-8" />
      </div>
    );
  }

  const navItems = [
    { label: 'Dashboard', icon: LayoutDashboard },
    { label: 'All Lectures', icon: BookOpen },
    { label: 'Played Lecture', icon: PlayCircle },
    { label: 'Shared Lecture', icon: Share2 },
    { label: 'Start New Lecture', icon: PlusCircle },
    { label: 'Q&A Section', icon: MessageSquare },
  ];

  const formatDate = (value) => {
    if (!value) return '—';
    try {
      return new Intl.DateTimeFormat('en-IN', {
        dateStyle: 'medium',
        timeStyle: 'short',
      }).format(new Date(value));
    } catch (error) {
      return value;
    }
  };

  const safeNumber = (value) => (typeof value === 'number' && !Number.isNaN(value) ? value : 0);
  const lectureMetrics = dashboardData?.lecture_metrics || {};

  const playedLectures = safeNumber(
    dashboardData?.played_lectures ??
      lectureMetrics?.played_lectures ??
      lectureMetrics?.performance?.total_lectures_played
  );
  const sharedLectures = safeNumber(
    dashboardData?.shared_lectures ??
      lectureMetrics?.shared_lectures ??
      lectureMetrics?.performance?.total_lectures_shared
  );
  const qaSessions = safeNumber(
    dashboardData?.qa_sessions ??
      lectureMetrics?.qa_sessions ??
      lectureMetrics?.performance?.total_qa_sessions
  );
  const totalLectures = safeNumber(
    dashboardData?.total_lectures ??
      lectureMetrics?.total_lectures ??
      lectureMetrics?.performance?.total_lectures_created
  );
  const pendingLectures = Math.max(totalLectures - playedLectures, 0);

  const chartSegments = [
    { label: 'Pending Lecture', value: pendingLectures, color: '#6C8BFF' },
    { label: 'Shared Lecture', value: Math.min(sharedLectures, Math.max(totalLectures, 0)), color: '#F5C84C' },
  ];

  const chartTotal = totalLectures > 0
    ? totalLectures
    : chartSegments.reduce((sum, segment) => sum + segment.value, 0);
  const hasChartData = chartTotal > 0;
  const normalizedTotal = hasChartData ? chartTotal : chartSegments.length;
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  let normalizedCumulative = 0;

  const donutLayers = chartSegments.map((segment) => {
    const normalizedValue = hasChartData ? segment.value : 1;
    const dash = (normalizedValue / normalizedTotal) * circumference;
    const dashOffset = circumference - ((normalizedCumulative + normalizedValue) / normalizedTotal) * circumference;
    normalizedCumulative += normalizedValue;

    return {
      ...segment,
      dashArray: `${dash} ${circumference}`,
      dashOffset,
    };
  });

  const legendItems = [
    { label: 'Pending Lecture', value: pendingLectures, color: '#6C8BFF' },
    { label: 'All Lecture', value: totalLectures, color: '#2D2E38' },
    { label: 'Shared Lecture', value: sharedLectures, color: '#F5C84C' },
  ];

  const statCards = [
    { label: 'Played Lecture', value: playedLectures },
    { label: 'Shared Lecture', value: sharedLectures },
    { label: 'Q & A Section', value: qaSessions },
  ];

  return (
    <div className="flex min-h-screen bg-[#050505] text-white">
      {/* Sidebar */}
      <aside className="hidden md:flex w-64 flex-col justify-between border-r border-white/5 bg-[#09090f] px-6 py-8">
        <div>
          <div className="mb-10">
            <p className="text-xs tracking-[0.4em] text-white/50">INAI</p>
            <p className="text-2xl font-semibold">VERSE</p>
          </div>
          <nav className="space-y-2">
            {navItems.map((item) => (
              <button
                key={item.label}
                type="button"
                onClick={() => handleNavClick(item)}
                className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm transition ${
                  selectedNav === item.label
                    ? 'bg-white/10 text-white shadow-inner shadow-white/10'
                    : 'text-white/50 hover:bg-white/5 hover:text-white'
                }`}
              >
                <item.icon className="h-5 w-5" />
                <span>{item.label}</span>
              </button>
            ))}
          </nav>
        </div>

        <button
          type="button"
          onClick={handleLogout}
          className="flex items-center gap-3 rounded-xl border border-white/10 px-4 py-3 text-sm text-white/70 transition hover:bg-white/5"
        >
          <LogOut className="h-5 w-5" />
          Logout
        </button>
      </aside>

      {/* Main Section */}
      <div className="flex-1 flex flex-col">
        <header className="flex items-center justify-between border-b border-white/5 bg-[#0c0c12] px-6 py-6">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-white/40">Dashboard</p>
            <h1 className="text-2xl font-semibold">Lecture Management</h1>
          </div>

          <div className="flex items-center gap-4">
            <button
              type="button"
              className="rounded-full border border-white/10 p-3 text-white/60 transition hover:text-white"
              aria-label="Toggle theme"
            >
              <Moon className="h-4 w-4" />
            </button>
            <button
              type="button"
              className="relative rounded-full border border-white/10 p-3 text-white/60 transition hover:text-white"
              aria-label="Notifications"
            >
              <Bell className="h-4 w-4" />
              <span className="absolute -top-1 -right-1 block h-2 w-2 rounded-full bg-red-500" />
            </button>
          </div>
        </header>

        <main className="flex-1 bg-[#050505] p-6 space-y-6">
          {selectedNav === 'Dashboard' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="rounded-3xl border border-white/5 bg-[#0d0d14] p-6 shadow-[0_20px_80px_rgba(0,0,0,0.45)]"
            >
              <div className="flex items-center justify-between border-b border-white/5 pb-6">
                <div>
                  <p className="text-sm uppercase tracking-[0.3em] text-white/40">Played Lecture</p>
                  <h2 className="text-3xl font-semibold">Lecture Management</h2>
                </div>
                <div className="text-right text-sm text-white/40">
                  <p>Welcome back</p>
                  <p className="text-white">{user?.name || 'Lecture Member'}</p>
                </div>
              </div>

              <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
                {statCards.map((stat) => (
                  <div
                    key={stat.label}
                    className="rounded-2xl border border-white/5 bg-[#11111a] px-6 py-5"
                  >
                    <p className="text-sm uppercase tracking-[0.2em] text-white/40">{stat.label}</p>
                    <p className="mt-2 text-4xl font-semibold">{stat.value}</p>
                  </div>
                ))}
              </div>

              <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-2">
                <div className="rounded-2xl border border-white/5 bg-[#11111a] p-6">
                  <h3 className="text-lg font-semibold">Total Lecture</h3>
                  <p className="text-sm text-white/50">Current overview</p>

                  <div className="mt-6 flex flex-col items-center gap-6 md:flex-row md:items-center">
                    <div className="relative h-48 w-48">
                      <svg width="100%" height="100%" viewBox="0 0 200 200">
                        <circle
                          cx="100"
                          cy="100"
                          r={radius}
                          fill="transparent"
                          stroke="#2D2E38"
                          strokeWidth="14"
                        />
                        {donutLayers.map((layer) => (
                          <circle
                            key={layer.label}
                            cx="100"
                            cy="100"
                            r={radius}
                            fill="transparent"
                            stroke={layer.color}
                            strokeWidth="14"
                            strokeLinecap="round"
                            strokeDasharray={layer.dashArray}
                            strokeDashoffset={layer.dashOffset}
                          />
                        ))}
                      </svg>
                      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                        <p className="text-sm text-white/40">Total</p>
                        <p className="text-3xl font-semibold">{totalLectures}</p>
                      </div>
                    </div>

                    <div className="flex-1 space-y-4">
                      {legendItems.map((item) => (
                        <div key={item.label} className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <span
                              className="h-3 w-3 rounded-full"
                              style={{ backgroundColor: item.color }}
                            />
                            <p className="text-sm text-white/70">{item.label}</p>
                          </div>
                          <p className="text-lg font-semibold">{item.value}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="rounded-2xl border border-white/5 bg-[#11111a] p-6">
                  <h3 className="text-lg font-semibold">Notes</h3>
                  <p className="text-sm text-white/50">
                    Logic integration is in progress. Metrics will update automatically once lecture data is available.
                  </p>
                  <div className="mt-6 rounded-2xl border border-dashed border-white/10 p-4 text-sm text-white/60">
                    <p>• Played, shared, and Q&A values default to zero until backend data arrives.</p>
                    <p>• Donut chart highlights Pending vs Shared distribution.</p>
                    <p>• Replace placeholder data once lecture APIs are ready.</p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {selectedNav === 'All Lectures' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="rounded-3xl border border-white/5 bg-[#0d0d14] p-6 shadow-[0_20px_80px_rgba(0,0,0,0.45)]"
            >
              <div className="flex flex-col gap-4 border-b border-white/5 pb-6 md:flex-row md:items-end md:justify-between">
                <div>
                  <p className="text-sm uppercase tracking-[0.3em] text-white/40">All Lectures</p>
                  <h2 className="text-3xl font-semibold">Generated Lecture Library</h2>
                </div>
                <div className="flex flex-col gap-3 md:flex-row">
                  <input
                    type="text"
                    value={lectureFilters.std}
                    onChange={(event) => handleFilterChange('std', event.target.value)}
                    placeholder="Filter by class (std)"
                    className="rounded-xl border border-white/10 bg-transparent px-4 py-2 text-sm text-white placeholder-white/40 focus:border-white/30 focus:outline-none"
                  />
                  <input
                    type="text"
                    value={lectureFilters.subject}
                    onChange={(event) => handleFilterChange('subject', event.target.value)}
                    placeholder="Filter by subject"
                    className="rounded-xl border border-white/10 bg-transparent px-4 py-2 text-sm text-white placeholder-white/40 focus:border-white/30 focus:outline-none"
                  />
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={handleApplyFilters}
                      className="rounded-xl border border-primary-400/50 bg-primary-500/20 px-4 py-2 text-sm font-medium text-white hover:bg-primary-500/30"
                    >
                      Apply
                    </button>
                    <button
                      type="button"
                      onClick={handleClearFilters}
                      className="rounded-xl border border-white/10 px-4 py-2 text-sm text-white/70 hover:bg-white/5"
                    >
                      Reset
                    </button>
                  </div>
                </div>
              </div>

              <div className="mt-6 rounded-2xl border border-white/5 bg-[#11111a]">
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[600px] text-left text-sm text-white/80">
                    <thead>
                      <tr className="border-b border-white/10 text-xs uppercase tracking-[0.3em] text-white/40">
                        <th className="px-4 py-3">Lecture</th>
                        <th className="px-4 py-3">Class</th>
                        <th className="px-4 py-3">Subject</th>
                        <th className="px-4 py-3">Created</th>
                        <th className="px-4 py-3">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {lecturesLoading && (
                        <tr>
                          <td colSpan={5} className="px-4 py-6 text-center text-white/50">
                            Loading lectures...
                          </td>
                        </tr>
                      )}
                      {!lecturesLoading && filteredLectures.length === 0 && (
                        <tr>
                          <td colSpan={5} className="px-4 py-6 text-center text-white/50">
                            No lectures found. Try generating a lecture from chapter management.
                          </td>
                        </tr>
                      )}
                      {!lecturesLoading && filteredLectures.length > 0 && filteredLectures.map((lecture) => (
                        <tr key={lecture.lecture_uid} className="border-b border-white/5 last:border-none">
                          <td className="px-4 py-3">
                            <p className="font-semibold text-white">{lecture.lecture_title}</p>
                            <p className="text-xs text-white/40">UID: {lecture.lecture_uid}</p>
                          </td>
                          <td className="px-4 py-3 text-white/70">{lecture.std || '—'}</td>
                          <td className="px-4 py-3 text-white/70">{lecture.subject || '—'}</td>
                          <td className="px-4 py-3 text-white/70">{formatDate(lecture.created_at)}</td>
                          <td className="px-4 py-3">
                            <div className="flex flex-wrap gap-2">
                              <a
                                href={lecture.lecture_json_url}
                                target="_blank"
                                rel="noreferrer"
                                className="rounded-lg border border-white/10 px-3 py-1 text-xs text-white/80 hover:bg-white/10"
                              >
                                View JSON
                              </a>
                              <button
                                type="button"
                                onClick={() => navigator.clipboard.writeText(`${window.location.origin}${lecture.lecture_json_url}`)
                                  .then(() => toast.success('Link copied to clipboard'))
                                  .catch(() => toast.error('Failed to copy link'))}
                                className="rounded-lg border border-white/10 px-3 py-1 text-xs text-white/60 hover:bg-white/10"
                              >
                                Copy Link
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </motion.div>
          )}
        </main>
      </div>
    </div>
  );
};

export default LectureDashboard;
