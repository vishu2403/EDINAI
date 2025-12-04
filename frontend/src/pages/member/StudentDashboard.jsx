/**
 * Student Dashboard - Dashboard for Student Management members
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { dashboardAPI, studentManagementAPI, getFileUrl } from '../../utils/api';
import {
  Bell,
  Download,
  FileUp,
  DollarSign,
  LogOut,
  Play,
  PlayCircle,
  Search,
  Share2,
  Upload,
  UserPlus,
  Users,
  Copy,
  ShieldCheck,
  AlertTriangle,
  Pencil,
  Trash2,
} from 'lucide-react';
import toast from 'react-hot-toast';

const StudentDashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState('dashboard');
  const [uploadFile, setUploadFile] = useState(null);
  const [downloadingTemplate, setDownloadingTemplate] = useState(false);
  const [uploadingRoster, setUploadingRoster] = useState(false);
  const [roster, setRoster] = useState([]);
  const [rosterLoading, setRosterLoading] = useState(false);
  const [studentProfiles, setStudentProfiles] = useState([]);
  const [profilesLoading, setProfilesLoading] = useState(false);
  const [duplicateSummary, setDuplicateSummary] = useState(null);
  const [duplicateReport, setDuplicateReport] = useState(null);
  const [uploadStats, setUploadStats] = useState(null);
  const [purchaseModalStudent, setPurchaseModalStudent] = useState(null);
  const [selectedPurchase, setSelectedPurchase] = useState(null);
  const [filterModalOpen, setFilterModalOpen] = useState(false);
  const [filterStep, setFilterStep] = useState('panel');
  const [classFilter, setClassFilter] = useState(null);
  const [divisionFilter, setDivisionFilter] = useState(null);
  const [filterOptions, setFilterOptions] = useState({ classes: [], divisions: [] });
  const [enrollmentSearch, setEnrollmentSearch] = useState('');

  useEffect(() => {
    loadDashboardData();
    fetchAvailableFilters();
  }, []);

  useEffect(() => {
    if (activeView === 'addStudent') {
      fetchRoster();
    }
    if (activeView === 'generateList') {
      fetchStudentProfiles();
    }
  }, [activeView]);

  const InfoSnippet = ({ label, value }) => (
    <div>
      <p className="text-[11px] uppercase tracking-[0.3em] text-white/40">{label}</p>
      <p className="text-sm font-semibold text-white mt-1">{value ?? '—'}</p>
    </div>
  );

  const formatINR = (value) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '₹0';
    return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(value);
  };

  const closePurchaseModal = () => {
    setPurchaseModalStudent(null);
    setSelectedPurchase(null);
  };

  const handleViewPurchases = (student) => {
    setPurchaseModalStudent(student);
    setSelectedPurchase(null);
  };

  const classOptions = filterOptions.classes || [];
  const divisionOptions = filterOptions.divisions || [];

  const applyFilters = async () => {
    await fetchStudentProfiles({ classFilter, divisionFilter });
    setFilterModalOpen(false);
    setFilterStep('panel');
  };

  const resetFilters = async () => {
    setClassFilter(null);
    setDivisionFilter(null);
    await fetchStudentProfiles({ classFilter: null, divisionFilter: null });
    await fetchAvailableFilters();
  };

  const handleClassSelect = (option) => {
    setClassFilter(option);
    setDivisionFilter(null);
    fetchAvailableFilters({ classFilter: option });
  };

  const handleEnrollmentSearch = async () => {
    const trimmed = enrollmentSearch.trim();
    setEnrollmentSearch(trimmed);
    await fetchStudentProfiles(trimmed ? { enrollmentFilter: trimmed } : { enrollmentFilter: null });
  };

  const clearEnrollmentSearch = async () => {
    if (!enrollmentSearch) return;
    setEnrollmentSearch('');
    await fetchStudentProfiles({ enrollmentFilter: null });
  };

  const renderProfileCards = () => {
    if (profilesLoading) {
      return <div className="py-8 text-center text-white/60">Loading student profiles…</div>;
    }

    if (!studentProfiles.length) {
      return (
        <div className="py-8 text-center text-white/40">
          No student has completed the profile form yet.
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {studentProfiles.map((student) => {
          const profileComplete = Boolean(student.profile_complete);
          const preferredFirstName = student.profile_first_name || student.roster_first_name || '';
          const fullName = [preferredFirstName, student.roster_last_name]
            .filter(Boolean)
            .join(' ')
            .trim() || '—';
          const stdLabel = student.std ? `STD ${student.std}` : '—';
          const divisionLabel = student.profile_division || student.roster_division || '—';
          const subjectLabel = student.class_stream || '—';
          const fatherName = student.father_name || student.roster_last_name || '—';
          const mobileNumber = student.mobile_number || '—';
          const parentsNumber = student.parents_number || '—';
          const email = student.email || '—';
          const avatarUrl = student.photo_path ? getFileUrl(student.photo_path) : null;
          const initials = fullName
            .split(' ')
            .filter(Boolean)
            .map((word) => word.charAt(0).toUpperCase())
            .slice(0, 2)
            .join('') || 'IN';

          return (
            <div
              key={`${student.enrollment_number}-${student.profile_id || 'roster'}`}
              className="rounded-[28px] border border-white/5 bg-[#0b0b0f] p-5 md:p-6"
            >
              <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-full border border-white/10 bg-white/5 flex items-center justify-center overflow-hidden">
                    {avatarUrl ? (
                      <img src={avatarUrl} alt={fullName} className="w-full h-full object-cover" />
                    ) : (
                      <span className="text-lg font-semibold">{initials}</span>
                    )}
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.4em] text-white/40">Name</p>
                    <p className="text-lg font-semibold text-white">{fullName}</p>
                    <div className="flex items-center gap-2 text-xs mt-1">
                      {profileComplete ? (
                        <>
                          <ShieldCheck className="w-4 h-4 text-emerald-400" />
                          <span className="text-emerald-200">Profile complete</span>
                        </>
                      ) : (
                        <>
                          <AlertTriangle className="w-4 h-4 text-orange-300" />
                          <span className="text-orange-200">Awaiting profile form</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
                <div className="flex flex-col items-start gap-3 md:items-end">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => toast.success(`Edit ${student.enrollment_number} coming soon`)}
                      className="inline-flex items-center gap-1 rounded-xl border border-white/10 px-3 py-1.5 text-xs text-white/80 hover:bg-white/10"
                    >
                      <Pencil className="w-3.5 h-3.5" />
                      Edit
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDeleteRosterStudent(student.enrollment_number)}
                      className="inline-flex items-center gap-1 rounded-xl border border-red-400/30 px-3 py-1.5 text-xs text-red-200 hover:bg-red-500/10"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      Delete
                    </button>
                  </div>
                  <p className="text-xs uppercase tracking-[0.4em] text-white/40">Password</p>
                  <div className="flex items-center gap-2 font-mono text-sm">
                    <span>{student.auto_password || '—'}</span>
                    {student.auto_password && (
                      <button
                        type="button"
                        onClick={() => navigator.clipboard.writeText(student.auto_password)}
                        className="inline-flex items-center gap-1 rounded-xl border border-white/10 px-2 py-1 text-xs text-white/80 hover:bg-white/10"
                      >
                        <Copy className="w-3.5 h-3.5" />
                        Copy
                      </button>
                    )}
                  </div>
                  {student.purchases?.length ? (
                    <button
                      type="button"
                      onClick={() => handleViewPurchases(student)}
                      className="inline-flex items-center justify-center w-full rounded-xl border border-emerald-400/30 px-3 py-1.5 text-xs font-semibold text-emerald-200 hover:bg-emerald-500/10"
                    >
                      View Purchase List
                    </button>
                  ) : (
                    <p className="text-[11px] uppercase tracking-[0.4em] text-white/30">No purchases yet</p>
                  )}
                </div>
              </div>

              <div className="mt-4 grid gap-4 md:grid-cols-3">
                <InfoSnippet label="Enrollment Number" value={student.enrollment_number} />
                <InfoSnippet label="Class" value={stdLabel} />
                <InfoSnippet label="Division" value={divisionLabel} />
                <InfoSnippet label="Subject" value={subjectLabel} />
                <InfoSnippet label="Father Name" value={fatherName} />
                <InfoSnippet label="Mobile Number" value={mobileNumber} />
                <InfoSnippet label="Parents Number" value={parentsNumber} />
                <InfoSnippet label="Email" value={email} />
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const loadDashboardData = async () => {
    try {
      const response = await dashboardAPI.getStudentDashboard();
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

  const fetchRoster = async () => {
    setRosterLoading(true);
    try {
      const response = await studentManagementAPI.fetchRoster();
      if (response.status) {
        setRoster(response.data?.students || []);
      }
    } catch (error) {
      if (activeView === 'generator') {
        toast.error('Failed to load student roster');
      }
    } finally {
      setRosterLoading(false);
    }
  };

  const fetchStudentProfiles = async (filters = {}) => {
    setProfilesLoading(true);
    try {
      const hasClassOverride = Object.prototype.hasOwnProperty.call(filters, 'classFilter');
      const hasDivisionOverride = Object.prototype.hasOwnProperty.call(filters, 'divisionFilter');
      const hasEnrollmentOverride = Object.prototype.hasOwnProperty.call(filters, 'enrollmentFilter');
      const response = await studentManagementAPI.fetchStudentProfiles({
        classFilter: hasClassOverride ? filters.classFilter ?? undefined : classFilter ?? undefined,
        divisionFilter: hasDivisionOverride ? filters.divisionFilter ?? undefined : divisionFilter ?? undefined,
        enrollmentFilter: hasEnrollmentOverride
          ? filters.enrollmentFilter ?? undefined
          : enrollmentSearch.trim() || undefined,
      });
      if (response.status) {
        setStudentProfiles(response.data?.students || []);
      }
    } catch (error) {
      if (activeView === 'generator') {
        toast.error('Failed to load student profiles');
      }
    } finally {
      setProfilesLoading(false);
    }
  };

  const fetchAvailableFilters = async (options = {}) => {
    try {
      const hasClassOverride = Object.prototype.hasOwnProperty.call(options, 'classFilter');
      const response = await studentManagementAPI.fetchStudentFilters(
        hasClassOverride ? { classFilter: options.classFilter ?? undefined } : {}
      );
      if (response.status) {
        const classes = response.data?.classes || [];
        const divisions = hasClassOverride ? response.data?.divisions || [] : [];
        setFilterOptions({ classes, divisions });
        setDivisionFilter((prev) => {
          if (!prev) return prev;
          return divisions.includes(prev) ? prev : null;
        });
      }
    } catch (error) {
      console.error('Failed to load filters', error);
    }
  };

  const renderStatsCard = ({ label, value, icon: Icon, action }) => {
    const cardContent = (
      <div className="rounded-2xl bg-[#101010] border border-white/5 p-5 space-y-3">
        <div className="flex items-center justify-between">
          <p className="text-xs uppercase tracking-widest text-white/60">{label}</p>
          <span className="p-2 rounded-xl bg-white/5">
            <Icon className="w-4 h-4" />
          </span>
        </div>
        <p className="text-3xl font-semibold tracking-tight">{value}</p>
      </div>
    );

    if (!action) return cardContent;

    return (
      <button
        type="button"
        onClick={action}
        className="text-left"
      >
        {cardContent}
      </button>
    );
  };

  const renderRosterRows = () => {
    if (rosterLoading) {
      return (
        <tr>
          <td colSpan={5} className="py-6 text-center text-white/60">
            Loading students…
          </td>
        </tr>
      );
    }

    if (!roster.length) {
      return (
        <tr>
          <td colSpan={5} className="py-6 text-center text-white/40">
            No students uploaded yet.
          </td>
        </tr>
      );
    }

    return roster.map((student) => (
      <tr key={student.enrollment_number} className="border-b border-white/5 last:border-0">
        <td className="py-3 px-4 font-semibold tracking-wide">{student.enrollment_number}</td>
        <td className="py-3 px-4">
          <p className="font-medium">{student.first_name} {student.last_name}</p>
          <p className="text-xs text-white/50">STD {student.std}{student.division ? ` • Div ${student.division}` : ''}</p>
        </td>
        <td className="py-3 px-4 font-mono text-sm">{student.auto_password}</td>
        <td className="py-3 px-4 text-sm text-white/70">
          {student.created_at ? new Date(student.created_at).toLocaleDateString('en-IN') : '—'}
        </td>
        <td className="py-3 px-4 text-sm text-white/60">
          <button
            type="button"
            onClick={() => navigator.clipboard.writeText(student.auto_password)}
            className="text-xs uppercase tracking-widest text-[#7cd2ff] hover:text-white"
          >
            Copy pass
          </button>
          <button
            type="button"
            onClick={() => handleDeleteRosterStudent(student.enrollment_number)}
            className="ml-4 inline-flex items-center gap-1 rounded-xl border border-red-400/30 px-3 py-1.5 text-[11px] uppercase tracking-[0.3em] text-red-200 hover:bg-red-500/10"
          >
            <Trash2 className="w-3 h-3" />
            Delete
          </button>
        </td>
      </tr>
    ));
  };

  const handleDeleteRosterStudent = async (enrollmentNumber) => {
    if (!enrollmentNumber) return;
    const confirmDelete = window.confirm(`Remove student ${enrollmentNumber}? This can't be undone.`);
    if (!confirmDelete) return;

    try {
      const response = await studentManagementAPI.deleteRosterStudent(enrollmentNumber);
      if (response.status) {
        toast.success(response.message || 'Student removed');
        await Promise.all([fetchRoster(), fetchStudentProfiles()]);
      } else {
        toast.error(response.message || 'Deletion failed');
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to delete student';
      toast.error(message);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/admin/login');
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-bg flex items-center justify-center">
        <div className="spinner w-8 h-8" />
      </div>
    );
  }

  const studentMetrics = dashboardData?.student_metrics || {};
  const totalStudents = studentMetrics.total_students ?? studentMetrics.total_chapters ?? 0;
  const watchedLectures = studentMetrics.progress?.completed_lectures ?? 0;
  const totalLectures = studentMetrics.total_lectures ?? studentMetrics.progress?.total_available_lectures ?? 0;
  const totalPaid = dashboardData?.package_info?.price ?? dashboardData?.admin_info?.package_price ?? 0;
  const lectureBalance = Math.max(totalLectures - watchedLectures, 0);
  const chartTotal = Math.max(watchedLectures + lectureBalance, 1);
  const studentPercent = Math.round((watchedLectures / chartTotal) * 100);
  const lecturePercent = 100 - studentPercent;
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const studentDash = (studentPercent / 100) * circumference;
  const lectureDash = (lecturePercent / 100) * circumference;

  const formatNumber = (value) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '0';
    if (value >= 1000) {
      return value.toLocaleString('en-IN');
    }
    return String(value);
  };

  const statsCards = [
    {
      label: 'Total Student',
      value: formatNumber(totalStudents),
      icon: Users,
      action: () => setActiveView('generateList'),
    },
    {
      label: 'Total Watched Lecture',
      value: formatNumber(watchedLectures),
      icon: PlayCircle,
    },
    {
      label: 'Total Paid',
      value:
        typeof totalPaid === 'number' && !Number.isNaN(totalPaid)
          ? totalPaid.toLocaleString('en-IN')
          : totalPaid || '—',
      icon: DollarSign,
    },
    {
      label: 'Add Student',
      value: 'Invite',
      icon: UserPlus,
      action: () => setActiveView('addStudent'),
    },
  ];

  const handleTemplateDownload = async () => {
    setDownloadingTemplate(true);
    try {
      const response = await studentManagementAPI.downloadTemplate();
      const blob = new Blob([response.data], {
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'student_roster_template.xlsx');
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      toast.success('Template downloaded');
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to download template';
      toast.error(message);
    } finally {
      setDownloadingTemplate(false);
    }
  };

  const handleUploadSubmit = async () => {
    if (!uploadFile) return;
    setUploadingRoster(true);
    setDuplicateSummary(null);
    setDuplicateReport(null);
    try {
      const response = await studentManagementAPI.uploadRoster(uploadFile);
      const { status: apiStatus, message: apiMessage, data } = response;
      const toastMessage = apiMessage || (apiStatus ? 'Student roster uploaded' : 'Upload skipped');
      (apiStatus ? toast.success : toast.error)(toastMessage);
      setUploadFile(null);
      setUploadStats({ added: data?.records_added || 0 });

      if (data?.duplicate_count) {
        setDuplicateSummary({ count: data.duplicate_count, rows: data.duplicates || [] });
        setDuplicateReport(data.duplicate_report || null);
      } else {
        setDuplicateSummary(null);
        setDuplicateReport(null);
      }

      if (apiStatus) {
        await Promise.all([fetchRoster(), fetchStudentProfiles()]);
      }
    } catch (error) {
      const message =
        error.response?.data?.detail || error.response?.data?.message || 'Failed to upload roster';
      toast.error(message);
    } finally {
      setUploadingRoster(false);
    }
  };

  const handleDuplicateDownload = () => {
    if (!duplicateReport) return;
    const { filename, content } = duplicateReport;
    try {
      const byteCharacters = atob(content);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i += 1) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], {
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename || 'duplicate_students.xlsx');
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      toast.success('Duplicate report downloaded');
    } catch (err) {
      console.error('Duplicate report download failed', err);
      toast.error('Could not download duplicate report');
    }
  };

  const renderDuplicatePreview = () => {
    if (!duplicateSummary?.rows?.length) return null;
    const sample = duplicateSummary.rows.slice(0, 3);
    return (
      <ul className="mt-3 space-y-1 text-xs text-white/70">
        {sample.map((row) => (
          <li key={`${row.row_number}-${row.enrollment_number}`} className="flex justify-between gap-2">
            <span>
              #{row.row_number || '—'} · {row.enrollment_number}
            </span>
            <span className="text-white/50">{row.reason}</span>
          </li>
        ))}
        {duplicateSummary.rows.length > sample.length && (
          <li className="text-white/50">+{duplicateSummary.rows.length - sample.length} more…</li>
        )}
      </ul>
    );
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white">
      <div className="flex min-h-screen">
        {/* Sidebar */}
        <aside className="w-64 bg-[#101010] border-r border-white/5 flex flex-col justify-between py-8 px-6">
          <div className="space-y-8">
            <div>
              <p className="text-2xl font-black tracking-[0.3em]">INAI</p>
              <p className="text-xs text-white/50 tracking-[0.4em] mt-1">verse</p>
            </div>
            <nav className="space-y-2 text-sm">
              <button
                className={`w-full text-left px-4 py-3 rounded-2xl border font-medium transition ${
                  activeView === 'dashboard'
                    ? 'bg-white/10 border-white/10 text-white'
                    : 'border-transparent text-white/60 hover:bg-white/5'
                }`}
                onClick={() => setActiveView('dashboard')}
              >
                Dashboard
              </button>
              <button
                className={`w-full text-left px-4 py-3 rounded-2xl border font-medium transition ${
                  activeView === 'addStudent'
                    ? 'bg-white/10 border-white/10 text-white'
                    : 'border-transparent text-white/60 hover:bg-white/5'
                }`}
                onClick={() => setActiveView('addStudent')}
              >
                Add Student
              </button>
              <button
                className={`w-full text-left px-4 py-3 rounded-2xl border font-medium transition ${
                  activeView === 'generateList'
                    ? 'bg-white/10 border-white/10 text-white'
                    : 'border-transparent text-white/60 hover:bg-white/5'
                }`}
                onClick={() => setActiveView('generateList')}
              >
                Generate Student List
              </button>
            </nav>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center gap-2 text-sm text-white/70 hover:text-white"
          >
            <LogOut className="w-4 h-4" />
            Logout
          </button>
        </aside>

        {/* Main content */}
        <main className="flex-1 p-8 space-y-8">
          {activeView === 'dashboard' && (
            <>
              {/* Header */}
              <div className="flex items-center justify-between bg-[#0f0f0f] border border-white/5 rounded-3xl px-6 py-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.6em] text-white/50">Dashboard</p>
                  <h1 className="text-2xl font-semibold mt-1">Student Management</h1>
                </div>
                <div className="flex items-center gap-3">
                  <button className="w-11 h-11 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center">
                    <Bell className="w-5 h-5" />
                  </button>
                  <div className="w-11 h-11 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center text-sm font-semibold">
                    {user?.name?.charAt(0)?.toUpperCase() || 'S'}
                  </div>
                </div>
              </div>

              {/* Stats row */}
              <motion.div
                className="grid grid-cols-1 md:grid-cols-4 gap-4"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
              >
                {statsCards.map((card) => (
                  <div key={card.label}>{renderStatsCard(card)}</div>
                ))}
              </motion.div>

              <section className="space-y-4">
                <h2 className="text-lg font-semibold tracking-tight">History</h2>
                <div className="rounded-3xl bg-[#0f0f0f] border border-white/5 p-6 max-w-lg">
                  <p className="text-sm uppercase tracking-[0.4em] text-white/50 mb-4">Total Lecture</p>
                  <div className="flex flex-col gap-6 items-center">
                    <div className="relative w-56 h-56">
                      <svg className="w-full h-full -rotate-90" viewBox="0 0 200 200">
                        <circle cx="100" cy="100" r="70" stroke="#1f1f1f" strokeWidth="18" fill="none" />
                        <circle
                          cx="100"
                          cy="100"
                          r="70"
                          stroke="#7cd2ff"
                          strokeWidth="14"
                          strokeLinecap="round"
                          fill="none"
                          strokeDasharray={`${(studentPercent / 100) * 2 * Math.PI * 70} ${2 * Math.PI * 70}`}
                          strokeDashoffset={0}
                        />
                        <circle
                          cx="100"
                          cy="100"
                          r="70"
                          stroke="#f9d264"
                          strokeWidth="14"
                          strokeLinecap="round"
                          fill="none"
                          strokeDasharray={`${((100 - studentPercent) / 100) * 2 * Math.PI * 70} ${2 * Math.PI * 70}`}
                          strokeDashoffset={2 * Math.PI * 70 - (studentPercent / 100) * 2 * Math.PI * 70}
                        />
                      </svg>
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <p className="text-4xl font-semibold">{studentPercent}%</p>
                        <span className="text-xs uppercase tracking-[0.4em] text-white/60">watched</span>
                      </div>
                    </div>

                    <div className="flex w-full gap-4 text-xs">
                      <div className="flex-1 rounded-2xl bg-white/5 border border-white/5 px-4 py-3 flex items-center gap-2">
                        <Play className="w-4 h-4 text-[#7cd2ff]" />
                        <div>
                          <p className="uppercase tracking-widest text-white/50">Student</p>
                          <p className="text-lg font-semibold">{formatNumber(watchedLectures)}</p>
                        </div>
                      </div>
                      <div className="flex-1 rounded-2xl bg-white/5 border border-white/5 px-4 py-3 flex items-center gap-2">
                        <Share2 className="w-4 h-4 text-[#f9d264]" />
                        <div>
                          <p className="uppercase tracking-widest text-white/50">Lecture</p>
                          <p className="text-lg font-semibold">{formatNumber(lectureBalance)}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </section>
            </>
          )}

          {activeView === 'addStudent' && (
            <>
              <div className="flex items-center justify-between bg-[#0f0f0f] border border-white/5 rounded-3xl px-6 py-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.6em] text-white/50">Generator</p>
                  <h1 className="text-2xl font-semibold mt-1">Add Student</h1>
                </div>
                <div className="flex items-center gap-3">
                  <button className="w-11 h-11 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center">
                    <Bell className="w-5 h-5" />
                  </button>
                  <div className="w-11 h-11 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center">
                    <UserPlus className="w-5 h-5" />
                  </div>
                </div>
              </div>

              <div className="flex justify-center">
                <div className="w-full max-w-3xl rounded-[32px] bg-[#111111] border border-white/5 px-8 py-10 space-y-8 shadow-[0_15px_45px_rgba(0,0,0,0.6)]">
                  <div className="text-center space-y-3">
                    <div className="w-16 h-16 mx-auto rounded-full bg-white/5 border border-white/10 flex items-center justify-center">
                      <FileUp className="w-7 h-7" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold">Excel File Upload</h3>
                      <p className="text-sm text-white/60">Upload your student data file to generate auto-passwords</p>
                    </div>
                  </div>

                  <div className="flex justify-center">
                    <button
                      type="button"
                      onClick={handleTemplateDownload}
                      disabled={downloadingTemplate}
                      className={`inline-flex items-center gap-2 px-6 py-2 border border-white/20 rounded-2xl bg-white/5 text-sm font-medium transition ${
                        downloadingTemplate ? 'opacity-60 cursor-not-allowed' : 'hover:bg-white/10'
                      }`}
                    >
                      <Download className="w-4 h-4" />
                      {downloadingTemplate ? 'Preparing…' : 'Download Template'}
                    </button>
                  </div>

                  <label
                    htmlFor="student-upload"
                    className="block rounded-2xl border-2 border-dashed border-[#7cd2ff]/60 bg-black/20 p-8 text-center cursor-pointer transition hover:border-[#7cd2ff]"
                  >
                    <input
                      id="student-upload"
                      type="file"
                      className="hidden"
                      accept=".xlsx,.xls,.csv"
                      onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                    />
                    <Upload className="w-8 h-8 mx-auto text-[#7cd2ff]" />
                    <p className="mt-3 text-sm font-semibold">Click to upload Excel file</p>
                    <p className="text-xs text-white/50">{uploadFile ? uploadFile.name : 'Drag & drop or browse files'}</p>
                  </label>

                  <div className="rounded-2xl bg-black/40 border border-white/5 p-4 text-sm space-y-1">
                    <p className="font-semibold">Auto-password format:</p>
                    <p className="text-white/70">First 4 letters of name + last 4 digits of enrollment</p>
                    <p className="text-white/50 text-xs">Example: Ravi 1234</p>
                  </div>

                  {uploadStats && (
                    <div className="rounded-2xl bg-white/5 border border-white/10 p-4 text-sm flex justify-between items-center">
                      <div>
                        <p className="font-semibold">Records added</p>
                        <p className="text-2xl font-bold">{uploadStats.added}</p>
                      </div>
                      {duplicateSummary?.count ? (
                        <p className="text-xs text-white/60">
                          {duplicateSummary.count} duplicate{duplicateSummary.count > 1 ? 's' : ''} skipped
                        </p>
                      ) : (
                        <p className="text-xs text-white/60">No duplicates detected</p>
                      )}
                    </div>
                  )}

                  {duplicateSummary?.count ? (
                    <div className="rounded-2xl bg-[#2a0f0f] border border-red-500/20 p-4 text-sm text-white">
                      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                        <div>
                          <p className="font-semibold">Duplicates detected</p>
                          <p className="text-white/70">
                            {duplicateSummary.count} row{duplicateSummary.count > 1 ? 's were' : ' was'} skipped to keep enrollment numbers unique.
                          </p>
                        </div>
                        {duplicateReport && (
                          <button
                            type="button"
                            onClick={handleDuplicateDownload}
                            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-white/20 bg-white/10 text-xs font-semibold hover:bg-white/20"
                          >
                            <Download className="w-4 h-4" /> Download duplicate Excel
                          </button>
                        )}
                      </div>
                      {renderDuplicatePreview()}
                    </div>
                  ) : null}

                  <button
                    type="button"
                    disabled={!uploadFile || uploadingRoster}
                    onClick={handleUploadSubmit}
                    className={`w-full py-3 rounded-2xl font-semibold flex items-center justify-center gap-2 ${
                      uploadFile && !uploadingRoster
                        ? 'bg-white text-black'
                        : 'bg-white/20 text-white/50 cursor-not-allowed'
                    }`}
                  >
                    <Upload className="w-4 h-4" />
                    {uploadingRoster ? 'Uploading…' : 'Submit File'}
                  </button>
                </div>
              </div>
              <section className="rounded-[32px] bg-[#0f0f0f] border border-white/5 px-6 py-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs uppercase tracking-[0.6em] text-white/40">Roster</p>
                    <h2 className="text-xl font-semibold">Generated Students</h2>
                  </div>
                  <p className="text-sm text-white/60">Auto-passwords ready to share</p>
                </div>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-left text-sm">
                    <thead>
                      <tr className="text-white/50">
                        <th className="py-3 px-4 font-semibold uppercase tracking-widest text-xs">Enrollment</th>
                        <th className="py-3 px-4 font-semibold uppercase tracking-widest text-xs">Student</th>
                        <th className="py-3 px-4 font-semibold uppercase tracking-widest text-xs">Auto Password</th>
                        <th className="py-3 px-4 font-semibold uppercase tracking-widest text-xs">Created</th>
                        <th className="py-3 px-4 font-semibold uppercase tracking-widest text-xs text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody>{renderRosterRows()}</tbody>
                  </table>
                </div>
              </section>
            </>
          )}

          {activeView === 'generateList' && (
            <>
              <div className="flex items-center justify-between bg-[#0f0f0f] border border-white/5 rounded-3xl px-6 py-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.6em] text-white/50">Profiles</p>
                  <h1 className="text-2xl font-semibold mt-1">Generate Student List</h1>
                </div>
                <div className="flex items-center gap-3 w-full max-w-lg">
                  <div className="flex items-center gap-2 flex-1 px-4 py-2 rounded-2xl bg-black/30 border border-white/10">
                    <Search className="w-4 h-4 text-white/50" />
                    <input
                      type="text"
                      className="bg-transparent text-sm flex-1 focus:outline-none"
                      placeholder="Search enrollment number"
                      value={enrollmentSearch}
                      onChange={(e) => setEnrollmentSearch(e.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          handleEnrollmentSearch();
                        }
                      }}
                    />
                    {enrollmentSearch ? (
                      <button
                        type="button"
                        onClick={clearEnrollmentSearch}
                        className="text-xs uppercase tracking-widest text-white/50 hover:text-white"
                      >
                        Clear
                      </button>
                    ) : null}
                  </div>
                  <button
                    type="button"
                    onClick={handleEnrollmentSearch}
                    className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-white/80 hover:bg-white/10"
                  >
                    Search
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setFilterModalOpen(true);
                      setFilterStep('panel');
                    }}
                    className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-white/80 hover:bg-white/10"
                  >
                    Filter Panel
                  </button>
                </div>
              </div>

              <section className="rounded-[32px] bg-[#0f0f0f] border border-white/5 px-6 py-6 space-y-4">
                <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                  <div>
                    <p className="text-xs uppercase tracking-[0.6em] text-white/40">Student list</p>
                    <h2 className="text-xl font-semibold">Profile Cards</h2>
                  </div>
                  <div className="flex flex-col gap-2 md:flex-row md:items-center md:gap-3">
                    {(classFilter || divisionFilter) ? (
                      <button
                        type="button"
                        onClick={resetFilters}
                        className="text-xs uppercase tracking-[0.4em] text-red-300 hover:text-red-100"
                      >
                        Clear Filters
                      </button>
                    ) : null}
                  </div>
                </div>
                {renderProfileCards()}
              </section>
            </>
          )}
        </main>
      </div>
      {filterModalOpen ? (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/80 px-4 py-8">
          <div className="w-full max-w-md rounded-[32px] bg-[#050505] border border-white/10 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <button
                type="button"
                onClick={() => {
                  if (filterStep === 'panel') {
                    setFilterModalOpen(false);
                  } else {
                    setFilterStep('panel');
                  }
                }}
                className="text-white/70 hover:text-white"
              >
                ←
              </button>
              <p className="text-sm uppercase tracking-[0.4em] text-white/50">Filter</p>
              <button
                type="button"
                onClick={() => setFilterModalOpen(false)}
                className="text-white/60 hover:text-white"
              >
                ✕
              </button>
            </div>
            {filterStep === 'panel' ? (
              <div className="rounded-2xl border border-white/10 bg-black/20">
                <button
                  type="button"
                  onClick={() => setFilterStep('class')}
                  className="flex w-full items-center justify-between px-4 py-3 border-b border-white/5 text-white"
                >
                  <span>Class</span>
                  <span>{classFilter || 'Any'}</span>
                </button>
                <button
                  type="button"
                  onClick={() => setFilterStep('division')}
                  className="flex w-full items-center justify-between px-4 py-3 text-white"
                >
                  <span>Division</span>
                  <span>{divisionFilter || 'Any'}</span>
                </button>
              </div>
            ) : null}
            {filterStep === 'class' ? (
              <div className="rounded-2xl border border-white/10 bg-black/30 p-4 space-y-4">
                <p className="text-sm text-white/70">Choose Class</p>
                {classOptions.length ? (
                  <div className="grid grid-cols-4 gap-2">
                    {classOptions.map((option) => (
                      <button
                        key={option}
                        type="button"
                        onClick={() => handleClassSelect(option)}
                        className={`flex h-10 items-center justify-center rounded-xl border text-sm font-semibold transition ${
                          classFilter === option ? 'border-white text-black bg-white' : 'border-white/10 text-white/70 hover:border-white/40'
                        }`}
                      >
                        {option}
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-white/50">No class data available yet.</p>
                )}
              </div>
            ) : null}
            {filterStep === 'division' ? (
              <div className="rounded-2xl border border-white/10 bg-black/30 p-4 space-y-4">
                <p className="text-sm text-white/70">Choose Division</p>
                {!classFilter ? (
                  <p className="text-xs text-white/50">Select a class first to see available divisions.</p>
                ) : divisionOptions.length ? (
                  <div className="grid grid-cols-4 gap-2">
                    {divisionOptions.map((option) => (
                      <button
                        key={option}
                        type="button"
                        onClick={() => setDivisionFilter(option)}
                        className={`flex h-10 items-center justify-center rounded-xl border text-sm font-semibold transition ${
                          divisionFilter === option ? 'border-white text-black bg-white' : 'border-white/10 text-white/70 hover-border-white/40'
                        }`}
                      >
                        {option}
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-white/50">No division data available yet.</p>
                )}
              </div>
            ) : null}
            <button
              type="button"
              onClick={applyFilters}
              className="w-full rounded-2xl bg-white/90 px-4 py-3 text-center text-sm font-semibold text-black"
            >
              Apply Filters
            </button>
          </div>
        </div>
      ) : null}
      {purchaseModalStudent ? (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/80 px-4 py-8">
          <div className="w-full max-w-3xl rounded-[32px] bg-[#0a0a0a] border border-white/10 p-8 space-y-6 relative">
            <button
              type="button"
              onClick={closePurchaseModal}
              className="absolute right-5 top-5 text-white/60 hover:text-white"
            >
              ✕
            </button>
            <div>
              <p className="text-xs uppercase tracking-[0.4em] text-white/40">Student Details</p>
              <h2 className="text-2xl font-semibold mt-2">{purchaseModalStudent.profile_first_name || purchaseModalStudent.roster_first_name || 'Student'}</h2>
              <div className="grid gap-2 mt-3 text-sm text-white/70">
                <span><strong>Enrollment Number:</strong> {purchaseModalStudent.enrollment_number}</span>
                <span>
                  <strong>Class:</strong> {purchaseModalStudent.class_stream || purchaseModalStudent.std ? `${purchaseModalStudent.class_stream || `STD ${purchaseModalStudent.std}`}${purchaseModalStudent.roster_division ? ` • Div ${purchaseModalStudent.roster_division}` : ''}` : '—'}
                </span>
              </div>
            </div>
            <div className="flex flex-col gap-3 md:flex-row">
              <button
                type="button"
                className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-center text-sm font-semibold"
              >
                History
              </button>
              <button
                type="button"
                className="flex-1 rounded-2xl border border-white px-4 py-3 text-center text-sm font-semibold text-black bg-white"
              >
                Old Purchase
              </button>
            </div>
            <div className="rounded-2xl bg-[#111111] border border-white/10 p-5 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.4em] text-white/40">Purchase History</p>
                <p className="text-white/80">Total Transaction Amount</p>
              </div>
              <p className="text-2xl font-semibold text-emerald-300">
                {formatINR((purchaseModalStudent.purchases || []).reduce((acc, item) => acc + (item.amount || 0), 0))}
              </p>
            </div>
            <div className="rounded-2xl bg-[#0d0d0f] border border-white/10 p-5 space-y-4 max-h-96 overflow-y-auto">
              {(purchaseModalStudent.purchases || []).length ? (
                purchaseModalStudent.purchases.map((purchase) => {
                  const status = purchase.status || 'Pending';
                  const statusCompleted = status.toLowerCase() === 'completed';
                  return (
                    <div
                      key={`${purchase.transaction_id}-${purchase.date}-${purchase.time}`}
                      className="rounded-2xl border border-white/5 bg-black/30 p-4"
                    >
                      <div className="grid gap-1 text-sm text-white/80">
                        <span><strong>Date:</strong> {purchase.date || '—'}</span>
                        <span><strong>Time:</strong> {purchase.time || '—'}</span>
                        <span><strong>Transaction Type:</strong> {purchase.transaction_type || '—'}</span>
                        <span><strong>Amount:</strong> {formatINR(purchase.amount || 0)}</span>
                        <span>
                          <strong>Status:</strong>{' '}
                          <span className={statusCompleted ? 'text-emerald-300' : 'text-orange-300'}>{status}</span>
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={() => setSelectedPurchase(purchase)}
                        className="mt-3 inline-flex items-center gap-2 rounded-xl border border-white/20 px-4 py-2 text-xs font-semibold hover:bg-white/10"
                      >
                        View Details
                      </button>
                    </div>
                  );
                })
              ) : (
                <p className="text-center text-white/50 text-sm">No purchase records found.</p>
              )}
            </div>
          </div>
        </div>
      ) : null}
      {selectedPurchase ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 px-4 py-8">
          <div className="w-full max-w-md rounded-[28px] bg-[#0f0f0f] border border-white/10 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Transaction Details</h3>
              <button
                type="button"
                onClick={() => setSelectedPurchase(null)}
                className="text-white/60 hover:text-white"
              >
                ✕
              </button>
            </div>
            <div className="space-y-2 text-sm text-white/80">
              <p><strong>Transaction ID:</strong> {selectedPurchase.transaction_id || '—'}</p>
              <p><strong>Date &amp; Time:</strong> {selectedPurchase.date || '—'} {selectedPurchase.time ? `• ${selectedPurchase.time}` : ''}</p>
              <p><strong>Payment Method:</strong> {selectedPurchase.payment_method || '—'}</p>
              <p><strong>Transaction Type:</strong> {selectedPurchase.transaction_type || '—'}</p>
              <p><strong>Description:</strong> {selectedPurchase.description || '—'}</p>
            </div>
            <div className="rounded-2xl bg-black/30 border border-white/10 p-4 text-center">
              <p className="text-xs uppercase tracking-[0.4em] text-white/50">Total Amount Paid</p>
              <p className="text-2xl font-semibold text-emerald-300 mt-1">{formatINR(selectedPurchase.amount || 0)}</p>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default StudentDashboard;
