/**
 * File Uploader Component
 * Handles drag & drop file uploads with preview and validation
 */

import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, File, Image } from 'lucide-react';
import clsx from 'clsx';
import toast from 'react-hot-toast';

const FileUploader = ({
  onFilesChange,
  accept = 'image/*',
  multiple = false,
  maxSize = 5 * 1024 * 1024, // 5MB
  maxFiles = 1,
  className = '',
  label = 'Upload Files',
  description = 'Drag and drop files here, or click to select',
}) => {
  const [files, setFiles] = useState([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const validateFile = (file) => {
    if (file.size > maxSize) {
      toast.error(`File "${file.name}" is too large. Maximum size is ${maxSize / (1024 * 1024)}MB`);
      return false;
    }
    return true;
  };

  const handleFiles = (newFiles) => {
    const validFiles = Array.from(newFiles).filter(validateFile);
    
    if (multiple) {
      const totalFiles = files.length + validFiles.length;
      if (totalFiles > maxFiles) {
        toast.error(`Maximum ${maxFiles} files allowed`);
        return;
      }
      const updatedFiles = [...files, ...validFiles];
      setFiles(updatedFiles);
      onFilesChange(updatedFiles);
    } else {
      if (validFiles.length > 0) {
        setFiles(validFiles);
        onFilesChange(validFiles);
      }
    }
  };

  const removeFile = (index) => {
    const updatedFiles = files.filter((_, i) => i !== index);
    setFiles(updatedFiles);
    onFilesChange(updatedFiles);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const droppedFiles = e.dataTransfer.files;
    handleFiles(droppedFiles);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleInputChange = (e) => {
    const selectedFiles = e.target.files;
    if (selectedFiles) {
      handleFiles(selectedFiles);
    }
  };

  const getFilePreview = (file) => {
    if (file.type.startsWith('image/')) {
      return URL.createObjectURL(file);
    }
    return null;
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className={clsx('space-y-4', className)}>
      {/* Upload Area */}
      <div
        className={clsx(
          'file-upload-area',
          {
            'dragover': isDragOver,
          }
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={accept}
          multiple={multiple}
          onChange={handleInputChange}
          className="hidden"
        />
        
        <Upload className="w-12 h-12 text-dark-text-muted mx-auto mb-4" />
        
        <h3 className="text-lg font-medium text-dark-text-primary mb-2">
          {label}
        </h3>
        
        <p className="text-dark-text-secondary text-sm mb-4">
          {description}
        </p>
        
        <div className="text-xs text-dark-text-muted">
          <p>Supported formats: {accept}</p>
          <p>Maximum size: {maxSize / (1024 * 1024)}MB per file</p>
          {multiple && <p>Maximum files: {maxFiles}</p>}
        </div>
      </div>

      {/* File Previews */}
      <AnimatePresence>
        {files.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-3"
          >
            <h4 className="text-sm font-medium text-dark-text-primary">
              Selected Files ({files.length})
            </h4>
            
            <div className="space-y-2">
              {files.map((file, index) => {
                const preview = getFilePreview(file);
                
                return (
                  <motion.div
                    key={`${file.name}-${index}`}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className="flex items-center space-x-3 p-3 bg-dark-card rounded-lg border border-dark-border"
                  >
                    {/* File Icon/Preview */}
                    <div className="flex-shrink-0">
                      {preview ? (
                        <img
                          src={preview}
                          alt={file.name}
                          className="w-12 h-12 object-cover rounded-lg"
                        />
                      ) : (
                        <div className="w-12 h-12 bg-gray-700 rounded-lg flex items-center justify-center">
                          <File className="w-6 h-6 text-gray-400" />
                        </div>
                      )}
                    </div>
                    
                    {/* File Info */}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-dark-text-primary truncate">
                        {file.name}
                      </p>
                      <p className="text-xs text-dark-text-muted">
                        {formatFileSize(file.size)}
                      </p>
                    </div>
                    
                    {/* Remove Button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeFile(index);
                      }}
                      className="p-1 text-gray-400 hover:text-red-400 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default FileUploader;
