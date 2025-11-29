/**
 * All Members Page - Admin can view and manage all team members
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { memberAPI } from '../../utils/api';
import Button from '../../components/ui/Button';
import { ArrowLeft, Users, Edit, Trash2, MoreVertical, UserPlus, Download } from 'lucide-react';
import toast from 'react-hot-toast';

const AllMembersPage = () => {
  const navigate = useNavigate();
  const [members, setMembers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    loadMembers();
  }, [filter]);

  const loadMembers = async () => {
    try {
      const params = filter !== 'all' ? { work_type: filter } : {};
      const response = await memberAPI.getMembers(params);
      if (response.status) {
        setMembers(response.data.members || []);
      }
    } catch (error) {
      console.error('Load members error:', error);
      toast.error('Failed to load members');
    } finally {
      setLoading(false);
    }
  };

  const handleToggleStatus = async (memberId) => {
    try {
      const response = await memberAPI.toggleMemberStatus(memberId);
      if (response.status) {
        toast.success('Member status updated');
        loadMembers();
      }
    } catch (error) {
      console.error('Toggle status error:', error);
      toast.error('Failed to update member status');
    }
  };

  const handleDelete = async (memberId) => {
    if (!confirm('Are you sure you want to delete this member?')) return;

    try {
      const response = await memberAPI.deleteMember(memberId);
      if (response.status) {
        toast.success('Member deleted successfully');
        loadMembers();
      }
    } catch (error) {
      console.error('Delete member error:', error);
      toast.error('Failed to delete member');
    }
  };

  const getWorkTypeColor = (workType) => {
    switch (workType) {
      case 'chapter':
        return 'bg-yellow-600';
      case 'student':
        return 'bg-blue-600';
      case 'lecture':
        return 'bg-green-600';
      default:
        return 'bg-gray-600';
    }
  };

  const getWorkTypeLabel = (workType) => {
    switch (workType) {
      case 'chapter':
        return 'Chapter Management';
      case 'student':
        return 'Student Management';
      case 'lecture':
        return 'Lecture Management';
      default:
        return workType;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-bg flex items-center justify-center">
        <div className="spinner w-8 h-8" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Header */}
      <div className="bg-dark-card border-b border-dark-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="medium"
              icon={ArrowLeft}
              onClick={() => navigate('/dev_admin')}
            />
            <div>
              <h1 className="text-xl font-bold text-dark-text-primary">Management List</h1>
              <p className="text-sm text-dark-text-muted">Manage your team members</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <Button
              variant="secondary"
              size="medium"
              icon={Download}
              onClick={() => toast.info('Export functionality coming soon')}
            >
              Export
            </Button>
            <Button
              variant="primary"
              size="medium"
              icon={UserPlus}
              onClick={() => navigate('/add-member')}
            >
              Add Member
            </Button>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Filter Tabs */}
        <div className="mb-6">
          <div className="flex space-x-1 bg-dark-card rounded-lg p-1">
            {[
              { key: 'all', label: 'All Management', count: members.length },
              { key: 'chapter', label: 'Chapter', count: members.filter(m => m.work_type === 'chapter').length },
              { key: 'student', label: 'Student', count: members.filter(m => m.work_type === 'student').length },
              { key: 'lecture', label: 'Lecture', count: members.filter(m => m.work_type === 'lecture').length },
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setFilter(tab.key)}
                className={`flex-1 px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                  filter === tab.key
                    ? 'bg-primary-600 text-white'
                    : 'text-dark-text-muted hover:text-dark-text-primary'
                }`}
              >
                {tab.label}
                <span className="ml-2 text-xs opacity-75">({tab.count})</span>
              </button>
            ))}
          </div>
        </div>

        {/* Members Grid */}
        {members.length === 0 ? (
          <motion.div
            className="card p-12 text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <Users className="w-16 h-16 text-dark-text-muted mx-auto mb-4" />
            <h3 className="text-lg font-medium text-dark-text-primary mb-2">
              No members found
            </h3>
            <p className="text-dark-text-muted mb-6">
              {filter === 'all' 
                ? "You haven't added any team members yet."
                : `No ${filter} members found.`
              }
            </p>
            <Button
              variant="primary"
              icon={UserPlus}
              onClick={() => navigate('/add-member')}
            >
              Add Your First Member
            </Button>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {members.map((member, index) => (
              <motion.div
                key={member.member_id}
                className="card p-6 hover:shadow-lg transition-all duration-300"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                {/* Member Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`w-12 h-12 ${getWorkTypeColor(member.work_type)} rounded-lg flex items-center justify-center`}>
                      <span className="text-white font-bold text-sm">
                        {member.name.charAt(0).toUpperCase()}
                      </span>
                    </div>
                    <div>
                      <h3 className="font-semibold text-dark-text-primary">
                        {member.name}
                      </h3>
                      <p className="text-sm text-dark-text-muted">
                        {member.role_id}
                      </p>
                    </div>
                  </div>
                  
                  <div className="relative">
                    <Button
                      variant="ghost"
                      size="small"
                      icon={MoreVertical}
                      className="!p-1"
                    />
                  </div>
                </div>

                {/* Member Details */}
                <div className="space-y-3">
                  <div>
                    <p className="text-xs text-dark-text-muted uppercase tracking-wide">Work Type</p>
                    <p className="text-sm text-dark-text-primary font-medium">
                      {getWorkTypeLabel(member.work_type)}
                    </p>
                  </div>
                  
                  <div>
                    <p className="text-xs text-dark-text-muted uppercase tracking-wide">Email</p>
                    <p className="text-sm text-dark-text-primary">
                      {member.email}
                    </p>
                  </div>
                  
                  <div>
                    <p className="text-xs text-dark-text-muted uppercase tracking-wide">Designation</p>
                    <p className="text-sm text-dark-text-primary">
                      {member.designation}
                    </p>
                  </div>

                  {member.phone_number && (
                    <div>
                      <p className="text-xs text-dark-text-muted uppercase tracking-wide">Phone</p>
                      <p className="text-sm text-dark-text-primary">
                        {member.phone_number}
                      </p>
                    </div>
                  )}

                  {/* Status */}
                  <div className="flex items-center justify-between pt-2">
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${
                        member.active ? 'bg-green-500' : 'bg-red-500'
                      }`} />
                      <span className={`text-xs font-medium ${
                        member.active ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {member.active ? 'Active' : 'Inactive'}
                      </span>
                    </div>
                    
                    <div className="text-xs text-dark-text-muted">
                      {member.last_login 
                        ? `Last login: ${new Date(member.last_login).toLocaleDateString()}`
                        : 'Never logged in'
                      }
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex space-x-2 mt-6 pt-4 border-t border-dark-border">
                  <Button
                    variant="ghost"
                    size="small"
                    fullWidth
                    icon={Edit}
                    onClick={() => toast.info('Edit functionality coming soon')}
                  >
                    Edit
                  </Button>
                  
                  <Button
                    variant={member.active ? 'secondary' : 'primary'}
                    size="small"
                    fullWidth
                    onClick={() => handleToggleStatus(member.member_id)}
                  >
                    {member.active ? 'Deactivate' : 'Activate'}
                  </Button>
                  
                  <Button
                    variant="danger"
                    size="small"
                    icon={Trash2}
                    onClick={() => handleDelete(member.member_id)}
                    className="!px-3"
                  />
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default AllMembersPage;
