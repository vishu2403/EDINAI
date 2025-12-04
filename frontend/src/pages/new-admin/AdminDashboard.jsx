import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { CreditCard, Users, Clock, BookOpen } from 'lucide-react';

import AdminShell from '../../components/new-admin/AdminShell.jsx';
import { useNewAdminAuth } from '../../contexts/NewAdminAuthContext.jsx';
import { newAdminMemberAPI, newAdminDashboardAPI } from '../../utils/newAdminApi.js';

const RING_STYLES = [
  { innerRadius: 110, outerRadius: 130, color: '#9BD1FF' },
  { innerRadius: 80, outerRadius: 100, color: '#F3D26C' },
  { innerRadius: 50, outerRadius: 70, color: '#C6A6FF' },
];

const AdminDashboard = () => {
  const { admin } = useNewAdminAuth();
  const adminId = admin?.admin_id;

  const [memberCount, setMemberCount] = useState(0);
  const [memberCountLoading, setMemberCountLoading] = useState(false);
  const [managementBreakdown, setManagementBreakdown] = useState({
    lecture: 0,
    chapter: 0,
    student: 0,
  });
  const [creditSummary, setCreditSummary] = useState({
    total: null,
    used: 0,
    remaining: null,
    postLimitGenerated: 0,
    activeDays: null,
  });
  const [creditLoading, setCreditLoading] = useState(false);

  useEffect(() => {
    if (!adminId) return;

    const fetchMemberData = async () => {
      try {
        setMemberCountLoading(true);
        const response = await newAdminMemberAPI.listMembers({ adminId, workType: 'all' });
        const count =
          response?.data?.count ??
          response?.count ??
          response?.data?.members?.length ??
          response?.members?.length ??
          0;
        const membersPayload =
          response?.data?.members ??
          response?.members ??
          response ??
          [];

        const normalizedMembers = Array.isArray(membersPayload) ? membersPayload : [];
        const breakdown = normalizedMembers.reduce(
          (acc, member) => {
            const type = member?.work_type?.toLowerCase();
            if (type?.includes('lecture')) acc.lecture += 1;
            else if (type?.includes('chapter')) acc.chapter += 1;
            else if (type?.includes('student')) acc.student += 1;
            return acc;
          },
          { lecture: 0, chapter: 0, student: 0 }
        );

        setMemberCount(count);
        setManagementBreakdown(breakdown);
      } catch (error) {
        console.error('Failed to load member count', error);
      } finally {
        setMemberCountLoading(false);
      }
    };

    fetchMemberData();
  }, [adminId]);

  useEffect(() => {
    if (!adminId) return;

    const fetchDashboardCredits = async () => {
      try {
        setCreditLoading(true);
        const response = await newAdminDashboardAPI.getAdminDashboard();
        const lectureCredits = response?.lecture_credits || response?.data?.lecture_credits || {};
        const accountStatus = response?.account_status || response?.data?.account_status || {};

        const total = typeof lectureCredits.total === 'number' ? lectureCredits.total : null;
        const used = typeof lectureCredits.used === 'number' ? lectureCredits.used : 0;
        const remaining = typeof lectureCredits.remaining === 'number'
          ? Math.max(lectureCredits.remaining, 0)
          : total !== null
            ? Math.max(total - used, 0)
            : null;

        setCreditSummary({
          total,
          used,
          remaining,
          postLimitGenerated: Math.max(lectureCredits.post_limit_generated || 0, 0),
          activeDays: typeof accountStatus.days_until_expiry === 'number'
            ? Math.max(accountStatus.days_until_expiry, 0)
            : null,
        });
      } catch (error) {
        console.error('Failed to load admin dashboard credits', error);
      } finally {
        setCreditLoading(false);
      }
    };

    fetchDashboardCredits();
  }, [adminId]);

  const pendingCredits = creditSummary.remaining;
  const creditsDisplay = creditLoading
    ? '...'
    : creditSummary.total !== null
      ? `${pendingCredits ?? 0}/${creditSummary.total}`
      : `${creditSummary.used}`;
  const creditsCaption = creditSummary.total !== null
    ? `Pending ${pendingCredits ?? 0}${typeof creditSummary.used === 'number' ? ` • Used ${creditSummary.used}` : ''}`
    : `Credits used ${creditSummary.used}`;
  const postLimitDisplay = creditLoading ? '...' : creditSummary.postLimitGenerated;
  const activeDaysDisplay = creditLoading
    ? '...'
    : creditSummary.activeDays !== null
      ? `${creditSummary.activeDays} days`
      : '—';

  const summaryCards = [
    { label: 'Management', value: memberCountLoading ? '...' : memberCount, icon: Users },
    { label: 'Total Credits', value: creditsDisplay, caption: creditsCaption, icon: CreditCard },
    { label: 'Count Of Lectures After Limit', value: postLimitDisplay, caption: 'Generated beyond plan limit', icon: BookOpen },
    { label: 'Active Subscription', value: activeDaysDisplay, caption: 'Remaining days', icon: Clock },
  ];

  const staffBreakdown = [
    { name: 'Lecture Management', value: managementBreakdown.lecture },
    { name: 'Chapter Management', value: managementBreakdown.chapter },
    { name: 'Student Management', value: managementBreakdown.student },
  ];

  const totalStaff = staffBreakdown.reduce((sum, curr) => sum + curr.value, 0) || 1;

  const managementSegments = staffBreakdown.map((item, index) => ({
    ...item,
    percent: Number(((item.value / totalStaff) * 100).toFixed(2)),
    ...(RING_STYLES[index] ?? RING_STYLES[RING_STYLES.length - 1]),
  }));

  return (
    <AdminShell title="Dashboard">
      <motion.div
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
          {summaryCards.map((card) => (
            <div key={card.label} className="bg-[#0E0F14] rounded-2xl p-5 border border-white/5 shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <p className="text-sm text-dark-text-muted uppercase tracking-wide">{card.label}</p>
                <card.icon className="w-5 h-5 text-primary-400" />
              </div>
              <p className="text-2xl font-semibold">{card.value}</p>
              {card.caption && (
                <p className="text-xs text-dark-text-secondary mt-1">{card.caption}</p>
              )}
            </div>
          ))}
      </motion.div>

      <motion.div
        className="bg-[#0E0F14] rounded-2xl p-6 border border-white/5 shadow-lg mt-8"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.05 }}
      >
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div>
            <h2 className="text-xl font-semibold">Total Management</h2>
            <p className="text-dark-text-muted text-sm mt-2">
              Overview of lecture, chapter, and student operations at a glance.
            </p>
          </div>
        </div>

        <div className="relative mt-8 h-64 md:h-72">
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="w-24 h-24 rounded-full border border-white/5 bg-[#0E0F14]" />
          </div>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              {managementSegments.map((segment) => (
                <Pie
                  key={segment.name}
                  data={[
                    { name: segment.name, value: segment.percent },
                    { name: 'remaining', value: 100 - segment.percent },
                  ]}
                  startAngle={90}
                  endAngle={450}
                  innerRadius={segment.innerRadius}
                  outerRadius={segment.outerRadius}
                  dataKey="value"
                  stroke="none"
                  strokeLinejoin="round"
                >
                  <Cell fill={segment.color} />
                  <Cell fill="#1B1F28" />
                </Pie>
              ))}
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-6 grid gap-3 sm:grid-cols-3">
          {managementSegments.map((segment) => (
            <div key={segment.name} className="flex items-center justify-between bg-[#10131A] px-4 py-3 rounded-2xl border border-white/5">
              <div className="flex items-center gap-2">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: segment.color }}
                />
                <span className="text-sm text-dark-text-secondary">{segment.name}</span>
              </div>
              <span className="text-sm font-medium">{segment.value}</span>
            </div>
          ))}
        </div>
      </motion.div>
    </AdminShell>
  );
};

export default AdminDashboard;