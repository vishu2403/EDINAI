/**
 * Admin Management Page - For Super Admins only
 * Allows super admins to create and manage other administrators
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { adminAPI } from '../../utils/api';
import Button from '../../components/ui/Button';
import { 
  Crown, 
  Plus, 
  Edit, 
  Trash2, 
  ArrowLeft,
  Calendar,
  Package,
  Shield,
  Check,
  X
} from 'lucide-react';
import toast from 'react-hot-toast';

const AdminManagement = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [admins, setAdmins] = useState([]);
  const [pagination, setPagination] = useState({ total: 0, skip: 0, limit: 0 });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is super admin
    if (!user?.is_super_admin) {
      navigate('/dev_admin');
      return;
    }
    
    loadAdmins();
  }, [user, navigate]);

  const loadAdmins = async () => {
    try {
      const response = await adminAPI.getAllAdmins();
      if (response.status) {
        setAdmins(response.data?.admins || []);
        setPagination({
          total: response.data?.total_count || 0,
          skip: response.data?.skip || 0,
          limit: response.data?.limit || 0,
        });
      }
    } catch (error) {
      console.error('Load admins error:', error);
      toast.error('Failed to load administrators');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteAdmin = async (adminId) => {
    if (!confirm('Are you sure you want to delete this administrator?')) {
      return;
    }

    try {
      const response = await adminAPI.deleteAdmin(adminId);
      if (response.status) {
        toast.success('Administrator deleted successfully');
        loadAdmins();
      }
    } catch (error) {
      console.error('Delete admin error:', error);
      toast.error('Failed to delete administrator');
    }
  };

  const getPackageName = (packageId) => {
    const packages = {
      'p1': 'Basic',
      'p2': 'Standard', 
      'p3': 'Premium'
    };
    return packages[packageId] || packageId;
  };

  const isExpired = (expiryDate) => {
    return new Date(expiryDate) < new Date();
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
            >
              Back
            </Button>
            <div>
              <h1 className="text-2xl font-bold text-dark-text-primary flex items-center gap-2">
                <Crown className="w-6 h-6 text-yellow-400" />
                Admin Management
              </h1>
              <p className="text-sm text-dark-text-muted">
                Manage all administrators in the system
              </p>
            </div>
          </div>
          
          <Button
            variant="primary"
            icon={Plus}
            onClick={() => navigate('/add-admin')}
          >
            Add Administrator
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <motion.div
            className="card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <span className="text-2xl font-bold text-dark-text-primary">
                {admins.length}
              </span>
            </div>
            <h3 className="font-medium text-dark-text-primary">Total Admins</h3>
            <p className="text-sm text-dark-text-muted">All administrators</p>
          </motion.div>

          <motion.div
            className="card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-green-600 rounded-lg flex items-center justify-center">
                <Check className="w-6 h-6 text-white" />
              </div>
              <span className="text-2xl font-bold text-dark-text-primary">
                {Array.isArray(admins) ? admins.filter(admin => admin.active).length : 0}
              </span>
            </div>
            <h3 className="font-medium text-dark-text-primary">Active Admins</h3>
            <p className="text-sm text-dark-text-muted">Currently active</p>
          </motion.div>

          <motion.div
            className="card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-red-600 rounded-lg flex items-center justify-center">
                <X className="w-6 h-6 text-white" />
              </div>
              <span className="text-2xl font-bold text-dark-text-primary">
                {Array.isArray(admins) ? admins.filter(admin => isExpired(admin.expiry_date)).length : 0}
              </span>
            </div>
            <h3 className="font-medium text-dark-text-primary">Expired</h3>
            <p className="text-sm text-dark-text-muted">Need renewal</p>
          </motion.div>
        </div>

        {/* Admins Table */}
        <motion.div
          className="card overflow-hidden"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.3 }}
        >
          <div className="p-6 border-b border-dark-border">
            <h2 className="text-xl font-semibold text-dark-text-primary">
              All Administrators
            </h2>
            <p className="text-sm text-dark-text-muted mt-1">
              Manage administrator accounts and permissions
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-dark-border bg-dark-bg/50">
                  <th className="text-left p-4 text-dark-text-primary font-medium">Admin</th>
                  <th className="text-left p-4 text-dark-text-primary font-medium">Package</th>
                  <th className="text-left p-4 text-dark-text-primary font-medium">Status</th>
                  <th className="text-left p-4 text-dark-text-primary font-medium">Expiry Date</th>
                  <th className="text-left p-4 text-dark-text-primary font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {admins.map((admin) => (
                  <tr key={admin.admin_id} className="border-b border-dark-border/50 hover:bg-dark-bg/30">
                    <td className="p-4">
                      <div className="flex items-center space-x-3">
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                          admin.is_super_admin 
                            ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' 
                            : 'bg-primary-600'
                        }`}>
                          {admin.is_super_admin ? (
                            <Crown className="w-5 h-5 text-white" />
                          ) : (
                            <span className="text-white font-medium">
                              {admin.name.charAt(0).toUpperCase()}
                            </span>
                          )}
                        </div>
                        <div>
                          <p className="font-medium text-dark-text-primary">
                            {admin.name}
                            {admin.is_super_admin && (
                              <span className="ml-2 text-xs bg-yellow-600/20 text-yellow-400 px-2 py-1 rounded">
                                SUPER
                              </span>
                            )}
                          </p>
                          <p className="text-sm text-dark-text-muted">{admin.email}</p>
                        </div>
                      </div>
                    </td>
                    <td className="p-4">
                      <div className="flex items-center space-x-2">
                        <Package className="w-4 h-4 text-primary-400" />
                        <span className="text-dark-text-primary">
                          {getPackageName(admin.package)}
                        </span>
                      </div>
                    </td>
                    <td className="p-4">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        admin.active
                          ? 'bg-green-600/20 text-green-400'
                          : 'bg-red-600/20 text-red-400'
                      }`}>
                        {admin.active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                    <td className="p-4">
                      <div className="flex items-center space-x-2">
                        <Calendar className="w-4 h-4 text-dark-text-muted" />
                        <span className={`text-sm ${
                          isExpired(admin.expiry_date) 
                            ? 'text-red-400' 
                            : 'text-dark-text-primary'
                        }`}>
                          {new Date(admin.expiry_date).toLocaleDateString()}
                        </span>
                      </div>
                    </td>
                    <td className="p-4">
                      <div className="flex items-center space-x-2">
                        <Button
                          variant="ghost"
                          size="small"
                          icon={Edit}
                          onClick={() => navigate(`/edit-admin/${admin.admin_id}`)}
                        />
                        {!admin.is_super_admin && (
                          <Button
                            variant="ghost"
                            size="small"
                            icon={Trash2}
                            className="text-red-400 hover:text-red-300"
                            onClick={() => handleDeleteAdmin(admin.admin_id)}
                          />
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {admins.length === 0 && (
            <div className="p-12 text-center">
              <Shield className="w-12 h-12 text-dark-text-muted mx-auto mb-4" />
              <p className="text-dark-text-muted">No administrators found</p>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default AdminManagement;
