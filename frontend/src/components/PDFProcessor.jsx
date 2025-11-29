/**
 * PDF Processor Component for INAI Education Management System
 * Handles PDF upload, OCR processing, and AI-powered topic splitting
 */

import React, { useState, useCallback } from 'react';
import { Upload, FileText, Brain, Loader2, CheckCircle, AlertCircle, Eye, Download } from 'lucide-react';
import toast from 'react-hot-toast';
import { processPDF, getPDFProcessingCapabilities } from '../utils/api';

const PDFProcessor = ({ onProcessingComplete }) => {
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  // Load capabilities on component mount
  React.useEffect(() => {
    loadCapabilities();
  }, []);

  const loadCapabilities = async () => {
    try {
      const caps = await getPDFProcessingCapabilities();
      setCapabilities(caps.capabilities);
    } catch (error) {
      console.error('Failed to load capabilities:', error);
    }
  };

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'application/pdf') {
        setFile(droppedFile);
      } else {
        toast.error('Please select a PDF file');
      }
    }
  }, []);

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.type === 'application/pdf') {
        setFile(selectedFile);
      } else {
        toast.error('Please select a PDF file');
      }
    }
  };

  const processFile = async () => {
    if (!file) {
      toast.error('Please select a PDF file first');
      return;
    }

    setProcessing(true);
    setResult(null);

    try {
      toast.loading('Processing PDF...', { id: 'pdf-processing' });
      
      const response = await processPDF(file);
      
      if (response.success) {
        setResult(response);
        toast.success('PDF processed successfully!', { id: 'pdf-processing' });
        
        // Notify parent component if callback provided
        if (onProcessingComplete) {
          onProcessingComplete(response);
        }
      } else {
        throw new Error(response.message || 'Processing failed');
      }
      
    } catch (error) {
      console.error('PDF processing error:', error);
      toast.error(error.message || 'Failed to process PDF', { id: 'pdf-processing' });
    } finally {
      setProcessing(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const downloadContent = (content, filename) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">PDF Processor</h2>
        <p className="text-gray-600">
          Upload PDF files for OCR processing and AI-powered topic splitting
        </p>
      </div>

      {/* Capabilities Info */}
      {capabilities && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Brain className="h-5 w-5 text-blue-600" />
            <span className="font-medium text-blue-900">Processing Capabilities</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">OCR:</span>
              <span className={`ml-1 ${capabilities.ocr_available ? 'text-green-600' : 'text-red-600'}`}>
                {capabilities.ocr_available ? 'Available' : 'Unavailable'}
              </span>
            </div>
            <div>
              <span className="text-gray-600">AI Analysis:</span>
              <span className={`ml-1 ${capabilities.ai_analysis_available ? 'text-green-600' : 'text-red-600'}`}>
                {capabilities.ai_analysis_available ? 'Available' : 'Unavailable'}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Max Size:</span>
              <span className="ml-1 text-gray-900">{capabilities.max_file_size_mb}MB</span>
            </div>
            <div>
              <span className="text-gray-600">AI Model:</span>
              <span className="ml-1 text-gray-900 text-xs">{capabilities.ai_model}</span>
            </div>
          </div>
        </div>
      )}

      {/* File Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive 
            ? 'border-blue-400 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {file ? (
          <div className="space-y-4">
            <FileText className="h-12 w-12 text-green-600 mx-auto" />
            <div>
              <p className="text-lg font-medium text-gray-900">{file.name}</p>
              <p className="text-sm text-gray-500">
                {formatFileSize(file.size)} • PDF Document
              </p>
            </div>
            <div className="flex justify-center space-x-4">
              <button
                onClick={processFile}
                disabled={processing}
                className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {processing ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Brain className="h-4 w-4 mr-2" />
                )}
                {processing ? 'Processing...' : 'Process PDF'}
              </button>
              <button
                onClick={() => setFile(null)}
                disabled={processing}
                className="inline-flex items-center px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 disabled:opacity-50"
              >
                Remove
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <Upload className="h-12 w-12 text-gray-400 mx-auto" />
            <div>
              <p className="text-lg font-medium text-gray-900">
                Drop your PDF file here, or{' '}
                <label className="text-blue-600 hover:text-blue-500 cursor-pointer">
                  browse
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </label>
              </p>
              <p className="text-sm text-gray-500 mt-1">
                Supports PDF files up to {capabilities?.max_file_size_mb || 50}MB
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Processing Results */}
      {result && (
        <div className="space-y-6">
          {/* Processing Summary */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <span className="font-medium text-green-900">Processing Complete</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Pages:</span>
                <span className="ml-1 text-gray-900">{result.metadata?.pages || 0}</span>
              </div>
              <div>
                <span className="text-gray-600">Words:</span>
                <span className="ml-1 text-gray-900">{result.processing_info?.word_count || 0}</span>
              </div>
              <div>
                <span className="text-gray-600">Characters:</span>
                <span className="ml-1 text-gray-900">{result.processing_info?.text_length || 0}</span>
              </div>
              <div>
                <span className="text-gray-600">Topics Split:</span>
                <span className="ml-1 text-gray-900">
                  {result.should_split_topics ? 'Yes' : 'No'}
                </span>
              </div>
            </div>
          </div>

          {/* AI Analysis Results */}
          {result.ai_analysis && result.ai_analysis.success && (
            <div className="bg-white border border-gray-200 rounded-lg">
              <div className="px-4 py-3 border-b border-gray-200">
                <h3 className="text-lg font-medium text-gray-900">
                  AI Analysis Results
                </h3>
                <p className="text-sm text-gray-500">
                  {result.ai_analysis.total_sections} sections identified
                </p>
              </div>
              
              <div className="divide-y divide-gray-200">
                {result.ai_analysis.sections.map((section, index) => (
                  <div key={index} className="p-4">
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="text-md font-medium text-gray-900">
                        {section.title}
                      </h4>
                      <div className="flex space-x-2">
                        <button
                          onClick={() => downloadContent(section.content, `${section.title}.txt`)}
                          className="text-blue-600 hover:text-blue-800"
                          title="Download section"
                        >
                          <Download className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-600 mb-3">
                      {section.summary}
                    </p>
                    
                    <div className="flex flex-wrap gap-2 mb-3">
                      {section.subtopics.map((subtopic, subIndex) => (
                        <span
                          key={subIndex}
                          className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                        >
                          {subtopic}
                        </span>
                      ))}
                    </div>
                    
                    <div className="text-xs text-gray-500">
                      {section.word_count} words • {section.char_count} characters
                    </div>
                    
                    <details className="mt-3">
                      <summary className="cursor-pointer text-sm text-blue-600 hover:text-blue-800">
                        View full content
                      </summary>
                      <div className="mt-2 p-3 bg-gray-50 rounded text-sm text-gray-700 max-h-40 overflow-y-auto">
                        {section.content}
                      </div>
                    </details>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Raw Text (if no AI analysis) */}
          {(!result.ai_analysis || !result.ai_analysis.success) && (
            <div className="bg-white border border-gray-200 rounded-lg">
              <div className="px-4 py-3 border-b border-gray-200">
                <h3 className="text-lg font-medium text-gray-900">
                  Extracted Text
                </h3>
              </div>
              <div className="p-4">
                <div className="bg-gray-50 rounded p-3 max-h-60 overflow-y-auto text-sm text-gray-700">
                  {result.extracted_text}
                </div>
                <div className="mt-3 flex justify-end">
                  <button
                    onClick={() => downloadContent(result.extracted_text, `${file.name}_extracted.txt`)}
                    className="inline-flex items-center px-3 py-1 text-sm text-blue-600 hover:text-blue-800"
                  >
                    <Download className="h-4 w-4 mr-1" />
                    Download Text
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PDFProcessor;
