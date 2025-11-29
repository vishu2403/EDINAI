/**
 * Admin Dashboard - Main dashboard for admins after onboarding
 * Shows metrics, member management, and system overview
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { dashboardAPI, chapterMaterialsAPI, getFileUrl } from '../../utils/api';
import Button from '../../components/ui/Button';
import InputField from '../../components/ui/InputField';
import FileUploader from '../../components/ui/FileUploader';
import { 
  Users, 
  UserPlus, 
  Settings, 
  LogOut, 
  Calendar,
  Package,
  TrendingUp,
  Menu,
  Bell,
  Search,
  Shield,
  Crown,
  FileText,
  Upload,
  Edit2,
  Trash2,
  XCircle
} from 'lucide-react';
import toast from 'react-hot-toast';

const BOARD_OPTIONS = ['GSEB', 'CBSE'];

const AdminDashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chapterMaterials, setChapterMaterials] = useState([]);
  const [materialsLoading, setMaterialsLoading] = useState(false);
  const [uploadingMaterial, setUploadingMaterial] = useState(false);
  const [chapterPdfFile, setChapterPdfFile] = useState([]);
  const [uploaderKey, setUploaderKey] = useState(0);
  const [editingMaterial, setEditingMaterial] = useState(null);
  const [chapterForm, setChapterForm] = useState({
    std: '',
    sem: '',
    board: BOARD_OPTIONS[0],
    subject: '',
    chapterNumber: ''
  });

  useEffect(() => {
    loadDashboardData();
    loadChapterMaterials();
  }, []);

  const loadDashboardData = async () => {
    try {
      const response = await dashboardAPI.getAdminDashboard();
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

  const loadChapterMaterials = async () => {
    try {
      setMaterialsLoading(true);
      const response = await chapterMaterialsAPI.listMaterials();
      if (response.status) {
        setChapterMaterials(response.data?.materials || []);
      }
    } catch (error) {
      console.error('Chapter materials load error:', error);
      toast.error('Failed to load chapter materials');
    } finally {
      setMaterialsLoading(false);
    }
  };

  const handleChapterFormChange = (field) => (event) => {
    const value = event.target.value;
    setChapterForm((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const resetChapterMaterialForm = () => {
    setChapterForm({
      std: '',
      sem: '',
      board: BOARD_OPTIONS[0],
      subject: '',
      chapterNumber: ''
    });
    setChapterPdfFile([]);
    setUploaderKey((prev) => prev + 1);
    setEditingMaterial(null);
  };

  const handleChapterMaterialUpload = async (event) => {
    event.preventDefault();

    const { std, sem, board, subject, chapterNumber } = chapterForm;
    if (!std.trim() || !sem.trim() || !board.trim() || !subject.trim() || !chapterNumber.trim()) {
      toast.error('Please fill in all chapter details.');
      return;
    }

    if (!editingMaterial && !chapterPdfFile.length) {
      toast.error('Please select a PDF file to upload.');
      return;
    }

    setUploadingMaterial(true);
    try {
      const formData = new FormData();
      formData.append('std', std.trim());
      formData.append('sem', sem.trim());
      formData.append('board', board.trim());
      formData.append('subject', subject.trim());
      formData.append('chapter_number', chapterNumber.trim());
      if (chapterPdfFile.length) {
        formData.append('pdf_file', chapterPdfFile[0]);
      }

      const isEditing = Boolean(editingMaterial);
      const response = isEditing
        ? await chapterMaterialsAPI.updateMaterial(editingMaterial.id, formData)
        : await chapterMaterialsAPI.uploadMaterial(formData);

      if (response.status) {
        toast.success(isEditing ? 'Chapter material updated successfully.' : 'Chapter material uploaded successfully.');
        resetChapterMaterialForm();
        await loadChapterMaterials();
      }
    } catch (error) {
      console.error('Chapter material upload error:', error);
      const message = error.response?.data?.detail || 'Failed to upload chapter material';
      toast.error(message);
    } finally {
      setUploadingMaterial(false);
    }
  };

  const handleChapterFileChange = (files) => {
    setChapterPdfFile(files || []);
  };

  const handleEditMaterial = (material) => {
    setEditingMaterial(material);
    setChapterForm({
      std: material.std || '',
      sem: material.sem || '',
      board: material.board || BOARD_OPTIONS[0],
      subject: material.subject || '',
      chapterNumber: material.chapter_number || '',
    });
    setChapterPdfFile([]);
    setUploaderKey((prev) => prev + 1);
  };

  const handleCancelEdit = () => {
    resetChapterMaterialForm();
  };

  const handleDeleteMaterial = async (materialId) => {
    const confirmed = window.confirm('Are you sure you want to delete this material? This action cannot be undone.');
    if (!confirmed) {
      return;
    }

    try {
      const response = await chapterMaterialsAPI.deleteMaterial(materialId);
      if (response.status) {
        toast.success('Chapter material deleted successfully.');
        if (editingMaterial?.id === materialId) {
          resetChapterMaterialForm();
        }
        await loadChapterMaterials();
      }
    } catch (error) {
      console.error('Chapter material delete error:', error);
      const message = error.response?.data?.detail || 'Failed to delete chapter material';
      toast.error(message);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B';
    if (!bytes) return '-';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-bg flex items-center justify-center">
        <div className="spinner w-8 h-8" />
      </div>
    );
  }

  const stats = dashboardData?.member_statistics || {};
  const packageInfo = dashboardData?.package_info || {};
  const accountStatus = dashboardData?.account_status || {};

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Mobile Menu Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 bg-dark-card border-r border-dark-border transform transition-transform duration-300 ease-in-out lg:translate-x-0 ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center justify-center h-16 border-b border-dark-border">
            <h1 className="text-2xl font-bold text-gradient">INAI</h1>
          </div>

          {/* User Info */}
          <div className="p-4 border-b border-dark-border">
            <div className="flex items-center space-x-3 mb-6">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                user?.is_super_admin ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' : 'bg-primary-600'
              }`}>
                {user?.is_super_admin ? (
                  <Crown className="w-5 h-5 text-white" />
                ) : (
                  <span className="text-white font-medium">
                    {user?.name?.charAt(0)?.toUpperCase() || 'A'}
                  </span>
                )}
              </div>
              <div>
                <p className="font-medium text-dark-text-primary">{user?.name}</p>
                <p className="text-xs text-dark-text-muted">
                  {user?.is_super_admin ? 'Super Administrator' : 'Administrator'}
                </p>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4 space-y-2">
            <Button
              variant="ghost"
              size="medium"
              fullWidth
              icon={TrendingUp}
              className="justify-start"
            >
              Management
            </Button>
            
            {/* Super Admin Only Options */}
            {user?.is_super_admin && (
              <>
                <Button
                  variant="ghost"
                  size="medium"
                  fullWidth
                  icon={Crown}
                  className="justify-start text-yellow-400 hover:text-yellow-300"
                  onClick={() => navigate('/admin-management')}
                >
                  Admin Management
                </Button>
                
                <Button
                  variant="ghost"
                  size="medium"
                  fullWidth
                  icon={Shield}
                  className="justify-start text-purple-400 hover:text-purple-300"
                  onClick={() => navigate('/add-admin')}
                >
                  Add Administrator
                </Button>
              </>
            )}
            
            <Button
              variant="ghost"
              size="medium"
              fullWidth
              icon={UserPlus}
              className="justify-start"
              onClick={() => navigate('/add-member')}
            >
              Add Member
            </Button>
            
            <Button
              variant="ghost"
              size="medium"
              fullWidth
              icon={Users}
              className="justify-start"
              onClick={() => navigate('/all-members')}
            >
              All Members
            </Button>
            
            <Button
              variant="ghost"
              size="medium"
              fullWidth
              icon={Settings}
              className="justify-start"
            >
              Settings
            </Button>
          </nav>

          {/* Logout */}
          <div className="p-4 border-t border-dark-border">
            <Button
              variant="danger"
              size="medium"
              fullWidth
              icon={LogOut}
              onClick={handleLogout}
              className="justify-start"
            >
              Logout
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="lg:ml-64">
        {/* Top Bar */}
        <header className="bg-dark-card border-b border-dark-border px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden p-2 text-dark-text-muted hover:text-dark-text-primary"
              >
                <Menu className="w-6 h-6" />
              </button>
              
              <div>
                <h1 className="text-xl font-bold text-dark-text-primary flex items-center gap-2">
                  {user?.is_super_admin ? (
                    <>
                      <Crown className="w-5 h-5 text-yellow-400" />
                      Super Admin Dashboard
                    </>
                  ) : (
                    'Admin Dashboard'
                  )}
                </h1>
                <p className="text-sm text-dark-text-muted">
                  Welcome back, {user?.is_super_admin ? 'Super Administrator' : 'Administrator'} {user?.name}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <button className="p-2 text-dark-text-muted hover:text-dark-text-primary">
                <Search className="w-5 h-5" />
              </button>
              <button className="p-2 text-dark-text-muted hover:text-dark-text-primary relative">
                <Bell className="w-5 h-5" />
                <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></span>
              </button>
              <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-medium">
                  {user?.name?.charAt(0)?.toUpperCase() || 'A'}
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Dashboard Content */}
        <main className="p-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {/* Total Members */}
            <motion.div
              className="card p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 bg-primary-600 rounded-lg flex items-center justify-center">
                  <Users className="w-6 h-6 text-white" />
                </div>
                <span className="text-2xl font-bold text-dark-text-primary">
                  {stats.total_members || 0}
                </span>
              </div>
              <h3 className="font-medium text-dark-text-primary">Management</h3>
              <p className="text-sm text-dark-text-muted">Total team members</p>
            </motion.div>

            {/* Active Members */}
            <motion.div
              className="card p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.1 }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 bg-green-600 rounded-lg flex items-center justify-center">
                  <TrendingUp className="w-6 h-6 text-white" />
                </div>
                <span className="text-2xl font-bold text-dark-text-primary">
                  {stats.active_members || 0}
                </span>
              </div>
              <h3 className="font-medium text-dark-text-primary">Total Student</h3>
              <p className="text-sm text-dark-text-muted">Active members</p>
            </motion.div>

            {/* Pending Videos */}
            <motion.div
              className="card p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.2 }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 bg-yellow-600 rounded-lg flex items-center justify-center">
                  <Package className="w-6 h-6 text-white" />
                </div>
                <span className="text-2xl font-bold text-dark-text-primary">
                  40
                </span>
              </div>
              <h3 className="font-medium text-dark-text-primary">Pending Videos</h3>
              <p className="text-sm text-dark-text-muted">Videos in queue</p>
            </motion.div>

            {/* Active Subscription */}
            <motion.div
              className="card p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.3 }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 bg-purple-600 rounded-lg flex items-center justify-center">
                  <Calendar className="w-6 h-6 text-white" />
                </div>
                <span className="text-2xl font-bold text-dark-text-primary">
                  5
                </span>
              </div>
              <h3 className="font-medium text-dark-text-primary">Active Subscription</h3>
              <p className="text-sm text-dark-text-muted">Days remaining</p>
            </motion.div>
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Total Staff Chart */}
            <motion.div
              className="lg:col-span-2 card p-6"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4 }}
            >
              <h3 className="text-lg font-semibold text-dark-text-primary mb-6">
                Total Staff
              </h3>
              
              {/* Staff Chart (Circular Progress) */}
              <div className="flex items-center justify-center h-64">
                <div className="relative w-48 h-48">
                  {/* Circular Progress */}
                  <svg className="w-48 h-48 transform -rotate-90" viewBox="0 0 144 144">
                    <circle
                      cx="72"
                      cy="72"
                      r="60"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      className="text-gray-700"
                    />
                    <circle
                      cx="72"
                      cy="72"
                      r="60"
                      stroke="url(#gradient)"
                      strokeWidth="8"
                      fill="none"
                      strokeDasharray={`${2 * Math.PI * 60}`}
                      strokeDashoffset={`${2 * Math.PI * 60 * (1 - 0.7)}`}
                      strokeLinecap="round"
                      className="transition-all duration-1000"
                    />
                    <defs>
                      <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#3B82F6" />
                        <stop offset="100%" stopColor="#06B6D4" />
                      </linearGradient>
                    </defs>
                  </svg>
                  
                  {/* Center Text */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <span className="text-3xl font-bold text-dark-text-primary">
                        {stats.active_members || 0}
                      </span>
                      <p className="text-sm text-dark-text-muted">Total Staff</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Legend */}
              <div className="flex items-center justify-center space-x-8 mt-6">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-blue-500 rounded-full"></div>
                  <span className="text-sm text-dark-text-muted">Teachers</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-yellow-500 rounded-full"></div>
                  <span className="text-sm text-dark-text-muted">Students</span>
                </div>
              </div>
            </motion.div>

            {/* Management List */}
            <motion.div
              className="card p-6"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: 0.1 }}
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-dark-text-primary">
                  Management List
                </h3>
                <Button
                  variant="primary"
                  size="small"
                  onClick={() => navigate('/add-member')}
                >
                  Add Member
                </Button>
              </div>

              {/* Management Categories */}
              <div className="space-y-4">
                <div className="gradient-border">
                  <div className="gradient-border-content p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-yellow-600 rounded-lg flex items-center justify-center">
                          <span className="text-white font-bold text-sm">📚</span>
                        </div>
                        <div>
                          <h4 className="font-medium text-dark-text-primary">
                            Chapter Management
                          </h4>
                          <p className="text-sm text-dark-text-muted">
                            All Management
                          </p>
                        </div>
                      </div>
                      <span className="text-lg font-bold text-dark-text-primary">
                        {stats.chapter_members || 0}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="gradient-border">
                  <div className="gradient-border-content p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                          <span className="text-white font-bold text-sm">👨‍🎓</span>
                        </div>
                        <div>
                          <h4 className="font-medium text-dark-text-primary">
                            Student Management
                          </h4>
                          <p className="text-sm text-dark-text-muted">
                            Management
                          </p>
                        </div>
                      </div>
                      <span className="text-lg font-bold text-dark-text-primary">
                        {stats.student_members || 0}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="gradient-border">
                  <div className="gradient-border-content p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-green-600 rounded-lg flex items-center justify-center">
                          <span className="text-white font-bold text-sm">🎓</span>
                        </div>
                        <div>
                          <h4 className="font-medium text-dark-text-primary">
                            Lecture Management
                          </h4>
                          <p className="text-sm text-dark-text-muted">
                            Management for
                          </p>
                        </div>
                      </div>
                      <span className="text-lg font-bold text-dark-text-primary">
                        {stats.lecture_members || 0}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="gradient-border">
                  <div className="gradient-border-content p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center">
                          <Users className="w-5 h-5 text-white" />
                        </div>
                        <div>
                          <h4 className="font-medium text-dark-text-primary">
                            All Members
                          </h4>
                          <p className="text-sm text-dark-text-muted">
                            View all team members
                          </p>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="small"
                        onClick={() => navigate('/all-members')}
                      >
                        View
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Chapter Material Upload */}
          <motion.div
            className="card p-6 mt-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6 mb-6">
              <div>
                <h3 className="text-lg font-semibold text-dark-text-primary">
                  Upload Chapter Material (PDF)
                </h3>
                <p className="text-sm text-dark-text-muted">
                  Add chapter-wise study materials to share with your students and team members.
                </p>
              </div>
            </div>

            {editingMaterial && (
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 p-4 mb-6 border border-primary-500/30 rounded-lg bg-primary-500/5">
                <div>
                  <p className="text-sm font-medium text-primary-200">Editing existing material</p>
                  <p className="text-sm text-dark-text-muted">
                    Std {editingMaterial.std} • Sem {editingMaterial.sem} • {editingMaterial.board} • {editingMaterial.subject} • {editingMaterial.chapter_number}
                  </p>
                  <p className="text-xs text-dark-text-muted mt-1">Update the details and optionally upload a new PDF to replace the existing one.</p>
                </div>
                <button
                  type="button"
                  onClick={handleCancelEdit}
                  className="flex items-center gap-2 px-3 py-2 rounded-md text-red-300 hover:text-red-200 border border-red-500/20 hover:border-red-500/40 transition"
                >
                  <XCircle className="w-4 h-4" />
                  Cancel Edit
                </button>
              </div>
            )}

            <form onSubmit={handleChapterMaterialUpload} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <InputField
                  label="Standard"
                  placeholder="e.g. 10"
                  value={chapterForm.std}
                  onChange={handleChapterFormChange('std')}
                  required
                />
                <InputField
                  label="Semester"
                  placeholder="e.g. 1"
                  value={chapterForm.sem}
                  onChange={handleChapterFormChange('sem')}
                  required
                />
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-dark-text-primary">
                    Board
                    <span className="text-red-500 ml-1">*</span>
                  </label>
                  <select
                    className="input-primary"
                    value={chapterForm.board}
                    onChange={handleChapterFormChange('board')}
                    required
                  >
                    {BOARD_OPTIONS.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </div>
                <InputField
                  label="Subject"
                  placeholder="e.g. Mathematics"
                  value={chapterForm.subject}
                  onChange={handleChapterFormChange('subject')}
                  required
                  containerClassName="md:col-span-2 lg:col-span-2"
                />
                <InputField
                  label="Chapter Number"
                  placeholder="e.g. Chapter 4"
                  value={chapterForm.chapterNumber}
                  onChange={handleChapterFormChange('chapterNumber')}
                  required
                  containerClassName="md:col-span-2 lg:col-span-1"
                />
              </div>

              <FileUploader
                key={uploaderKey}
                label="Chapter PDF"
                description="Upload a single PDF file (max 15MB)."
                accept="application/pdf"
                maxSize={15 * 1024 * 1024}
                multiple={false}
                onFilesChange={handleChapterFileChange}
              />

              {editingMaterial && (
                <p className="text-xs text-dark-text-muted -mt-4">Leave the file selection empty to keep the current PDF.</p>
              )}

              <div className="flex justify-end">
                <Button
                  type="submit"
                  variant="primary"
                  icon={Upload}
                  loading={uploadingMaterial}
                >
                  {editingMaterial ? 'Update Material' : 'Upload PDF'}
                </Button>
              </div>
            </form>

            <div className="mt-8">
              <h4 className="text-md font-semibold text-dark-text-primary mb-4 flex items-center gap-2">
                <FileText className="w-4 h-4 text-primary-400" />
                Uploaded Materials
              </h4>
              {materialsLoading ? (
                <div className="flex justify-center py-6">
                  <div className="spinner w-6 h-6" />
                </div>
              ) : chapterMaterials.length ? (
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead className="text-left text-dark-text-muted">
                      <tr className="border-b border-dark-border">
                        <th className="py-3 pr-4 font-medium">Standard</th>
                        <th className="py-3 pr-4 font-medium">Semester</th>
                        <th className="py-3 pr-4 font-medium">Board</th>
                        <th className="py-3 pr-4 font-medium">Subject</th>
                        <th className="py-3 pr-4 font-medium">Chapter</th>
                        <th className="py-3 pr-4 font-medium">File</th>
                        <th className="py-3 pr-4 font-medium">Uploaded On</th>
                        <th className="py-3 pr-4 font-medium text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {chapterMaterials.map((material) => (
                        <tr
                          key={material.id}
                          className="border-b border-dark-border/60 last:border-b-0"
                        >
                          <td className="py-3 pr-4 text-dark-text-primary">{material.std}</td>
                          <td className="py-3 pr-4 text-dark-text-primary">{material.sem}</td>
                          <td className="py-3 pr-4 text-dark-text-primary">{material.board}</td>
                          <td className="py-3 pr-4 text-dark-text-primary">{material.subject}</td>
                          <td className="py-3 pr-4 text-dark-text-primary">{material.chapter_number}</td>
                          <td className="py-3 pr-4">
                            <a
                              href={getFileUrl(material.file_path)}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-2 text-primary-400 hover:text-primary-300"
                            >
                              <FileText className="w-4 h-4" />
                              View PDF ({formatFileSize(material.file_size)})
                            </a>
                          </td>
                          <td className="py-3 pr-4 text-dark-text-secondary">
                            {material.created_at ? new Date(material.created_at).toLocaleString() : '-'}
                          </td>
                          <td className="py-3 pr-4">
                            <div className="flex items-center justify-end gap-2">
                              <button
                                type="button"
                                onClick={() => handleEditMaterial(material)}
                                className="p-2 rounded-md bg-dark-card border border-dark-border/60 text-primary-300 hover:text-primary-100 hover:border-primary-500/40 transition"
                                title="Edit material"
                              >
                                <Edit2 className="w-4 h-4" />
                              </button>
                              <button
                                type="button"
                                onClick={() => handleDeleteMaterial(material.id)}
                                className="p-2 rounded-md bg-dark-card border border-dark-border/60 text-red-400 hover:text-red-200 hover:border-red-500/40 transition"
                                title="Delete material"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-sm text-dark-text-muted">
                  No chapter materials uploaded yet.
                </p>
              )}
            </div>
          </motion.div>
        </main>
      </div>
    </div>
  );
};

export default AdminDashboard;
