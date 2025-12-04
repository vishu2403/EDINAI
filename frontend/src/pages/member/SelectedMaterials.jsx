/**
 * Selected Materials - Page for managing selected chapter materials
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { chapterMaterialsAPI, getFileUrl, generateTopicPDF } from '../../utils/api';
import Button from '../../components/ui/Button';
import InputField from '../../components/ui/InputField';
import {
  BookOpen,
  FileText,
  ArrowLeft,
  Plus,
  Trash2,
  CheckSquare,
  Square,
  ChevronDown,
  BookPlus,
  Users,
  Filter,
  Sparkles,
  Loader2,
  Eye,
  X,
} from 'lucide-react';
import toast from 'react-hot-toast';

const SelectedMaterials = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  
  // Get selected materials from navigation state
  const { selectedMaterials = [], filters = {} } = location.state || {};
  
  const [materials, setMaterials] = useState(selectedMaterials);
  const [selectedChapters, setSelectedChapters] = useState(new Set());
  const [topicFilters, setTopicFilters] = useState({});
  const [availableTopics, setAvailableTopics] = useState({});
  const [selectedTopics, setSelectedTopics] = useState({}); // New: Track selected topics per material
  const [loadingTopics, setLoadingTopics] = useState(false);
  const [generatingPDF, setGeneratingPDF] = useState(false); // New: PDF generation loading
  const [mergedLectureTitle, setMergedLectureTitle] = useState('');
  const [showMergePanel, setShowMergePanel] = useState(false);
  const [previewMaterial, setPreviewMaterial] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [showTopicPreview, setShowTopicPreview] = useState(false); // New: Topic-specific preview
  const [lectureLoading, setLectureLoading] = useState(false);

  useEffect(() => {
    if (materials.length === 0) {
      toast.error('No materials selected');
      navigate('/chapter-dashboard');
      return;
    }
    
    // Don't automatically extract topics - let user trigger it manually
    // Users can click "Extract Topics" button for individual materials
  }, [materials]);

  const extractTopicsFromMaterials = async () => {
    setLoadingTopics(true);
    try {
      const materialIds = materials.map(m => m.id);
      toast.loading(`Extracting topics from ${materialIds.length} materials... This may take up to 2 minutes.`, { id: 'topic-extraction' });
      
      const response = await chapterMaterialsAPI.extractTopics(materialIds);
      
      if (response.status) {
        // Handle new topics structure with detected language
        const topicsData = response.data.topics || {};
        const processedTopics = {};
        
        Object.keys(topicsData).forEach(materialId => {
          const materialData = topicsData[materialId];
          if (materialData && materialData.topics) {
            // New structure: {topics: [...], detected_language: "..."}
            processedTopics[materialId] = materialData.topics;
            console.log(`📄 Material ${materialId} detected language: ${materialData.detected_language}`);
          } else if (Array.isArray(materialData)) {
            // Old structure: just array of topics
            processedTopics[materialId] = materialData;
          }
        });
        
        setAvailableTopics(processedTopics);
        toast.success('Topics extracted successfully!', { id: 'topic-extraction' });
      } else {
        // Don't show fallback topics - let user know extraction failed
        toast.error('Topic extraction failed. Please try again.', { id: 'topic-extraction' });
        setAvailableTopics({});
      }
      
      // Initialize topic filters
      const initialFilters = {};
      materials.forEach(material => {
        initialFilters[material.id] = '';
      });
      setTopicFilters(initialFilters);
    } catch (error) {
      console.error('Topic extraction error:', error);
      
      // More specific error message
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error';
      toast.error(`Failed to extract topics: ${errorMessage}`, { id: 'topic-extraction' });
      
      // Don't show fallback topics - keep empty state
      setAvailableTopics({});
      
      const initialFilters = {};
      materials.forEach(material => {
        initialFilters[material.id] = '';
      });
      setTopicFilters(initialFilters);
    } finally {
      setLoadingTopics(false);
    }
  };

  const handleGenerateLecture = async (material) => {
    const selectedTopicIndices = Array.from(selectedTopics[material.id] || []);
    if (selectedTopicIndices.length === 0) {
      toast.error('Lecture generate કરવા માટે topics select કરો');
      return;
    }

    setLectureLoading(true);
    const toastId = `lecture-${material.id}`;
    toast.loading('Lecture generate થતું છે... થોડો સમય લાગી શકે.', { id: toastId, duration: 10000 });

    try {
      const response = await chapterMaterialsAPI.generateLectureFromTopics({
        materialId: material.id,
        topicIndices: selectedTopicIndices,
        options: {},
      });

      if (response.status && response.data?.lecture) {
        toast.success('Lecture તૈયાર છે!', { id: toastId });
        navigate('/lecture-editor', {
          state: {
            lecture: response.data.lecture,
            material: response.data.material,
          },
        });
      } else {
        toast.error('Lecture generate થઈ શક્યું નથી', { id: toastId });
      }
    } catch (error) {
      const message = error.response?.data?.detail || error.message || 'Lecture generate રાખતી વખતે ત્રુટિ આવી';
      toast.error(message, { id: toastId });
    } finally {
      setLectureLoading(false);
    }
  };

  const handleChapterSelect = (materialId) => {
    setSelectedChapters(prev => {
      const newSet = new Set(prev);
      if (newSet.has(materialId)) {
        newSet.delete(materialId);
      } else {
        newSet.add(materialId);
      }
      return newSet;
    });
  };

  const handleSelectAll = () => {
    if (selectedChapters.size === materials.length) {
      setSelectedChapters(new Set());
    } else {
      setSelectedChapters(new Set(materials.map(m => m.id)));
    }
  };

  const handleTopicFilterChange = (materialId, topic) => {
    setTopicFilters(prev => ({
      ...prev,
      [materialId]: topic
    }));
  };

  // New: Handle individual topic selection
  const handleTopicSelect = (materialId, topicIndex) => {
    setSelectedTopics(prev => {
      const materialTopics = prev[materialId] || new Set();
      const newTopics = new Set(materialTopics);
      
      if (newTopics.has(topicIndex)) {
        newTopics.delete(topicIndex);
      } else {
        newTopics.add(topicIndex);
      }
      
      return {
        ...prev,
        [materialId]: newTopics
      };
    });
  };

  // New: Select all topics for a material
  const handleSelectAllTopics = (materialId) => {
    const topics = availableTopics[materialId] || [];
    const allSelected = selectedTopics[materialId]?.size === topics.length;
    
    setSelectedTopics(prev => ({
      ...prev,
      [materialId]: allSelected ? new Set() : new Set(topics.map((_, index) => index))
    }));
  };

  // New: Preview selected topics only
  const handlePreviewSelectedTopics = (material) => {
    const selectedTopicIndices = selectedTopics[material.id] || new Set();
    if (selectedTopicIndices.size === 0) {
      toast.error('કૃપા કરીને પહેલા topics select કરો');
      return;
    }
    
    setPreviewMaterial({
      ...material,
      selectedTopicIndices: Array.from(selectedTopicIndices)
    });
    setShowTopicPreview(true);
  };

  // New: Generate PDF with selected topics
  const handleGenerateTopicPDF = async (material) => {
    const selectedTopicIndices = selectedTopics[material.id] || new Set();
    if (selectedTopicIndices.size === 0) {
      toast.error('કૃપા કરીને પહેલા topics select કરો');
      return;
    }

    setGeneratingPDF(true);
    try {
      const response = await generateTopicPDF(material.id, Array.from(selectedTopicIndices));
      
      if (response.success) {
        // Use language-specific message from backend
        toast.success(`✅ ${response.message}`);
        
        // Open the generated PDF in new tab
        const pdfUrl = getFileUrl(response.file_path);
        window.open(pdfUrl, '_blank');
        
        // Close the preview modal
        setShowTopicPreview(false);
        
        // Log language info
        console.log(`📄 PDF generated in ${response.detected_language} language`);
      } else {
        toast.error('Failed to generate PDF');
      }
    } catch (error) {
      console.error('PDF generation error:', error);
      toast.error('Failed to generate PDF: ' + (error.response?.data?.detail || error.message));
    } finally {
      setGeneratingPDF(false);
    }
  };

  const handleCreateMergedLecture = async () => {
    if (selectedChapters.size === 0) {
      toast.error('Please select at least one chapter');
      return;
    }
    
    if (!mergedLectureTitle.trim()) {
      toast.error('Please enter a title for the merged lecture');
      return;
    }

    try {
      const selectedMaterialsData = materials.filter(m => selectedChapters.has(m.id));
      const topicFiltersData = {};
      selectedMaterialsData.forEach(material => {
        if (topicFilters[material.id]) {
          topicFiltersData[material.id] = topicFilters[material.id];
        }
      });

      const lectureData = {
        title: mergedLectureTitle.trim(),
        material_ids: Array.from(selectedChapters),
        topic_filters: topicFiltersData,
        filters: filters // Include original search filters
      };

      const response = await chapterMaterialsAPI.createMergedLecture(lectureData);
      
      if (response.status) {
        toast.success('Merged lecture created successfully!');
        setShowMergePanel(false);
        setMergedLectureTitle('');
        setSelectedChapters(new Set());
        
        // Navigate back to dashboard or to a new page showing the created lecture
        navigate('/chapter-dashboard', { 
          state: { 
            message: 'Merged lecture created successfully!',
            lectureId: response.data.lecture_id 
          } 
        });
      } else {
        toast.error('Failed to create merged lecture');
      }
    } catch (error) {
      console.error('Create merged lecture error:', error);
      const message = error.response?.data?.detail || 'Failed to create merged lecture';
      toast.error(message);
    }
  };

  const filteredMaterials = materials.filter(material => {
    const selectedTopic = topicFilters[material.id];
    if (!selectedTopic || selectedTopic === '') return true;
    
    const topics = availableTopics[material.id];
    if (!topics) return false;
    
    // Handle both old format (array of strings) and new format (array of objects)
    return topics.some(topic => {
      const topicTitle = typeof topic === 'string' ? topic : topic.title;
      return topicTitle === selectedTopic;
    });
  });

  const handlePreviewPDF = (material) => {
    setPreviewMaterial(material);
    setShowPreview(true);
  };

  const closePreview = () => {
    setShowPreview(false);
    setPreviewMaterial(null);
  };

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Header */}
      <header className="bg-dark-card border-b border-dark-border px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="small"
              icon={ArrowLeft}
              onClick={() => navigate('/chapter-dashboard')}
              className="text-dark-text-muted hover:text-dark-text-primary"
            >
              Back
            </Button>
            <div className="w-10 h-10 bg-primary-600 rounded-full flex items-center justify-center">
              <BookPlus className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-dark-text-primary">Add Chapter Management</h1>
              <p className="text-sm text-dark-text-muted">Merge selected chapters into comprehensive lectures</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="small"
              icon={Sparkles}
              onClick={extractTopicsFromMaterials}
              disabled={loadingTopics}
              className="text-primary-400 hover:text-primary-300"
            >
              {loadingTopics ? 'Extracting...' : 'Extract All Topics'}
            </Button>
            <Button
              variant="secondary"
              size="small"
              onClick={() => setShowMergePanel(!showMergePanel)}
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              size="small"
              onClick={() => setShowMergePanel(true)}
              disabled={selectedChapters.size === 0}
            >
              Next
            </Button>
          </div>
        </div>
      </header>

      <main className="p-6 space-y-6">
        {/* Merge Chapter Section */}
        <motion.div
          className="card p-6"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
        >
          <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6 mb-6">
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-dark-text-primary mb-2">Merge Chapter</h3>
              <p className="text-sm text-dark-text-muted">
                Select Chapters From Multiple Books To Merge Into A Complete Lecture
              </p>
            </div>
            
            <div className="flex items-center gap-3">
              <Button
                variant="ghost"
                size="small"
                icon={selectedChapters.size === materials.length ? CheckSquare : Square}
                onClick={handleSelectAll}
                className="text-dark-text-muted hover:text-dark-text-primary"
              >
                {selectedChapters.size === materials.length ? 'Deselect All' : 'Select All'}
              </Button>
              {loadingTopics && (
                <div className="flex items-center gap-2 text-xs text-dark-text-muted">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Extracting topics...
                </div>
              )}
            </div>
          </div>

          {/* Materials List */}
          <div className="space-y-4">
            {filteredMaterials.map((material) => (
              <motion.div
                key={material.id}
                className={`rounded-xl border p-5 transition-all ${
                  selectedChapters.has(material.id)
                    ? 'border-primary-500 bg-primary-600/10'
                    : 'border-dark-border bg-dark-card/70'
                }`}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2 }}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start gap-4 flex-1">
                    <div 
                      className="flex-shrink-0 cursor-pointer"
                      onClick={() => handleChapterSelect(material.id)}
                    >
                      {selectedChapters.has(material.id) ? (
                        <CheckSquare className="w-5 h-5 text-primary-400 mt-1" />
                      ) : (
                        <Square className="w-5 h-5 text-dark-text-muted hover:text-dark-text-primary mt-1" />
                      )}
                    </div>
                    
                    <div className="flex-1 space-y-3">
                      <div>
                        <h4 className="text-base font-semibold text-dark-text-primary">
                          {material.subject}
                        </h4>
                        <p className="text-sm text-dark-text-muted">
                          by {material.author || 'Unknown Author'} • {material.chapter_number ? `${material.chapter_number} chapters` : 'Multiple chapters'}
                        </p>
                      </div>
                      
                      <div className="flex items-center gap-4 text-sm">
                        <span className="px-2 py-1 bg-dark-bg rounded text-dark-text-muted">
                          Class {material.std}
                        </span>
                        <span className="px-2 py-1 bg-dark-bg rounded text-dark-text-muted">
                          {material.board}
                        </span>
                        <span className="px-2 py-1 bg-dark-bg rounded text-dark-text-muted">
                          Sem {material.sem}
                        </span>
                      </div>

                      {/* Topic Filter Dropdown */}
                      {availableTopics[material.id] && availableTopics[material.id].length > 0 ? (
                        <div className="flex items-center gap-3">
                          <Filter className="w-4 h-4 text-dark-text-muted" />
                          <select
                            value={topicFilters[material.id] || ''}
                            onChange={(e) => handleTopicFilterChange(material.id, e.target.value)}
                            className="input-primary text-sm max-w-xs"
                          >
                            <option value="">All Topics</option>
                            {availableTopics[material.id].map((topic, index) => {
                              const topicTitle = typeof topic === 'string' ? topic : topic.title;
                              const topicSummary = typeof topic === 'object' ? topic.summary : '';
                              return (
                                <option key={index} value={topicTitle} title={topicSummary}>
                                  {topicTitle}
                                </option>
                              );
                            })}
                          </select>
                        </div>
                      ) : loadingTopics ? (
                        <div className="flex items-center gap-2 text-sm text-dark-text-muted">
                          <div className="w-4 h-4 border-2 border-primary-400 border-t-transparent rounded-full animate-spin"></div>
                          <span>Extracting topics...</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-3">
                          <div className="flex items-center gap-2 text-sm text-dark-text-muted">
                            <Filter className="w-4 h-4 text-dark-text-muted" />
                            <span>No topics extracted yet</span>
                          </div>
                          <button
                            onClick={async () => {
                              setLoadingTopics(true);
                              toast.loading('Starting topic extraction... This may take a few minutes for large PDFs.', { 
                                id: `topic-${material.id}`,
                                duration: 10000 
                              });
                              
                              // Add progress updates
                              const progressTimer = setInterval(() => {
                                toast.loading('Processing PDF content... Please wait.', { 
                                  id: `topic-${material.id}`,
                                  duration: 10000 
                                });
                              }, 30000); // Update every 30 seconds
                              
                              try {
                                const response = await chapterMaterialsAPI.extractTopics([material.id]);
                                if (response.status) {
                                  // Handle new topics structure
                                  const topicsData = response.data.topics || {};
                                  const processedTopics = {};
                                  
                                  Object.keys(topicsData).forEach(materialId => {
                                    const materialData = topicsData[materialId];
                                    if (materialData && materialData.topics) {
                                      processedTopics[materialId] = materialData.topics;
                                      console.log(`📄 Material ${materialId} detected language: ${materialData.detected_language}`);
                                    } else if (Array.isArray(materialData)) {
                                      processedTopics[materialId] = materialData;
                                    }
                                  });
                                  
                                  setAvailableTopics(prev => ({
                                    ...prev,
                                    ...processedTopics
                                  }));
                                  toast.success('Topics extracted!', { id: `topic-${material.id}` });
                                } else {
                                  toast.error('Failed to extract topics', { id: `topic-${material.id}` });
                                }
                              } catch (error) {
                                console.error('Topic extraction error:', error);
                                let errorMessage = 'Error extracting topics';
                                
                                if (error.response?.data?.message) {
                                  errorMessage = error.response.data.message;
                                } else if (error.message) {
                                  errorMessage = error.message;
                                } else if (!error.response) {
                                  errorMessage = 'Network error. Please check your connection and try again.';
                                }
                                
                                toast.error(errorMessage, { id: `topic-${material.id}` });
                              } finally {
                                clearInterval(progressTimer);
                                setLoadingTopics(false);
                              }
                            }}
                            className="px-3 py-1 text-xs bg-primary-600 text-white rounded hover:bg-primary-700 transition-colors"
                          >
                            Extract Topics
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <Button
                      variant="ghost"
                      size="small"
                      icon={Eye}
                      className="text-blue-400 hover:text-blue-300"
                      onClick={() => handlePreviewPDF(material)}
                      title="Preview PDF"
                    >
                    </Button>
                    <Button
                      variant="ghost"
                      size="small"
                      icon={Plus}
                      className="text-primary-400 hover:text-primary-300"
                    >
                    </Button>
                    <Button
                      variant="ghost"
                      size="small"
                      icon={Trash2}
                      className="text-red-400 hover:text-red-300"
                      onClick={() => {
                        setMaterials(prev => prev.filter(m => m.id !== material.id));
                        setSelectedChapters(prev => {
                          const newSet = new Set(prev);
                          newSet.delete(material.id);
                          return newSet;
                        });
                      }}
                    >
                    </Button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Add Book Button */}
          <div className="mt-6">
            <Button
              variant="ghost"
              size="small"
              icon={Plus}
              className="text-dark-text-muted hover:text-dark-text-primary border border-dashed border-dark-border hover:border-dark-border/80 w-full py-4"
            >
              Add Book
            </Button>
          </div>
        </motion.div>

        {/* Merged Lecture Panel */}
        {showMergePanel && (
          <motion.div
            className="card p-6"
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.35 }}
          >
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-dark-text-primary">Merged Lecture</h3>
                <p className="text-sm text-dark-text-muted">
                  {selectedChapters.size} chapter(s) selected
                </p>
              </div>

              <div className="p-4 bg-dark-bg/60 rounded-lg">
                <div className="flex items-center gap-2 text-sm text-dark-text-muted mb-2">
                  <Users className="w-4 h-4" />
                  Select chapters from different books to create a comprehensive lecture
                </div>
                
                <InputField
                  placeholder="Enter lecture title"
                  value={mergedLectureTitle}
                  onChange={(e) => setMergedLectureTitle(e.target.value)}
                  className="mb-4"
                />

                <Button
                  variant="primary"
                  size="small"
                  icon={Sparkles}
                  onClick={handleCreateMergedLecture}
                  className="w-full"
                  disabled={selectedChapters.size === 0 || !mergedLectureTitle.trim()}
                >
                  Create Merged Lecture
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </main>

      {/* PDF Preview Modal */}
      {showPreview && previewMaterial && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <motion.div
            className="bg-dark-card rounded-xl border border-dark-border w-full max-w-6xl h-full max-h-[90vh] flex flex-col"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b border-dark-border">
              <div className="flex items-center gap-3">
                <FileText className="w-5 h-5 text-primary-400" />
                <div>
                  <h3 className="text-lg font-semibold text-dark-text-primary">
                    {previewMaterial.subject}
                  </h3>
                  <p className="text-sm text-dark-text-muted">
                    Class {previewMaterial.std} • {previewMaterial.board} • Chapter {previewMaterial.chapter_number}
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="small"
                icon={X}
                onClick={closePreview}
                className="text-dark-text-muted hover:text-dark-text-primary"
              />
            </div>

            {/* PDF Viewer with Topics Panel */}
            <div className="flex-1 p-4 flex gap-4">
              {/* PDF Viewer */}
              <div className="flex-1 bg-dark-bg/60 rounded-lg border border-dark-border overflow-hidden">
                {previewMaterial.file_path ? (
                  <iframe
                    src={getFileUrl(previewMaterial.file_path)}
                    className="w-full h-full"
                    title={`Preview: ${previewMaterial.subject}`}
                    onError={(e) => {
                      console.error('PDF load error:', e);
                      toast.error('Failed to load PDF preview');
                    }}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center space-y-3">
                      <FileText className="w-12 h-12 text-dark-text-muted mx-auto" />
                      <p className="text-dark-text-muted">PDF preview not available</p>
                      <p className="text-xs text-dark-text-muted">
                        File path: {previewMaterial.file_name || 'Unknown'}
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Topics Panel */}
              <div className="w-80 bg-dark-card rounded-lg border border-dark-border flex flex-col">
                <div className="p-4 border-b border-dark-border">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <BookOpen className="w-5 h-5 text-primary-400" />
                      <h4 className="font-semibold text-dark-text-primary">Extracted Topics</h4>
                    </div>
                    {availableTopics[previewMaterial.id] && availableTopics[previewMaterial.id].length > 0 && (
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleSelectAllTopics(previewMaterial.id)}
                          className="text-xs px-2 py-1 bg-primary-500/20 text-primary-400 rounded hover:bg-primary-500/30 transition-colors"
                        >
                          {selectedTopics[previewMaterial.id]?.size === availableTopics[previewMaterial.id].length ? 'Deselect All' : 'Select All'}
                        </button>
                        <button
                          onClick={() => handleGenerateLecture(previewMaterial)}
                          disabled={lectureLoading || !selectedTopics[previewMaterial.id] || selectedTopics[previewMaterial.id].size === 0}
                          className="text-xs px-2 py-1 bg-purple-500/20 text-purple-300 rounded hover:bg-purple-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {lectureLoading ? 'Generating...' : 'Generate Lecture'}
                        </button>
                        <button
                          onClick={() => handlePreviewSelectedTopics(previewMaterial)}
                          disabled={!selectedTopics[previewMaterial.id] || selectedTopics[previewMaterial.id].size === 0}
                          className="text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded hover:bg-green-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          Preview Selected
                        </button>
                      </div>
                    )}
                  </div>
                  <p className="text-xs text-dark-text-muted mt-1">
                    AI-generated topics from this material • Select topics to preview individually
                  </p>
                </div>
                
                <div className="flex-1 p-4 overflow-y-auto max-h-[calc(90vh-200px)] scrollbar-thin scrollbar-thumb-primary-400/30 scrollbar-track-dark-bg/20">
                  {loadingTopics ? (
                    <div className="flex items-center justify-center h-32">
                      <div className="text-center space-y-2">
                        <div className="w-6 h-6 border-2 border-primary-400 border-t-transparent rounded-full animate-spin mx-auto"></div>
                        <p className="text-sm text-dark-text-muted">Extracting topics...</p>
                      </div>
                    </div>
                  ) : availableTopics[previewMaterial.id] && availableTopics[previewMaterial.id].length > 0 ? (
                    <div className="space-y-3 pr-2">
                      {availableTopics[previewMaterial.id].map((topic, index) => (
                        <div
                          key={index}
                          className={`p-4 rounded-lg border transition-colors group cursor-pointer ${
                            selectedTopics[previewMaterial.id]?.has(index)
                              ? 'bg-primary-500/10 border-primary-400/50'
                              : 'bg-dark-bg/40 border-dark-border/50 hover:border-primary-400/30'
                          }`}
                        >
                          <div className="flex items-start gap-3">
                            {/* Checkbox */}
                            <div 
                              className="flex-shrink-0 mt-1"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleTopicSelect(previewMaterial.id, index);
                              }}
                            >
                              {selectedTopics[previewMaterial.id]?.has(index) ? (
                                <CheckSquare className="w-5 h-5 text-primary-400" />
                              ) : (
                                <Square className="w-5 h-5 text-dark-text-muted hover:text-primary-400" />
                              )}
                            </div>
                            
                            {/* Topic Number */}
                            <div className="w-7 h-7 bg-primary-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                              <span className="text-xs font-medium text-primary-400">{index + 1}</span>
                            </div>
                            <div className="flex-1 min-w-0">
                              <h5 className="text-sm font-medium text-dark-text-primary group-hover:text-primary-400 transition-colors break-words">
                                {typeof topic === 'string' ? topic : topic.title}
                              </h5>
                              {typeof topic === 'object' && topic.summary && (
                                <p className="text-xs text-dark-text-muted mt-1 line-clamp-3 break-words">
                                  {topic.summary}
                                </p>
                              )}
                              {typeof topic === 'object' && topic.subtopics && topic.subtopics.length > 0 && (
                                <div className="mt-2 space-y-1">
                                  {topic.subtopics.map((subtopic, subIndex) => {
                                    const isObject = subtopic && typeof subtopic === 'object';
                                    const subtopicTitle = isObject ? subtopic.title : subtopic;
                                    const subtopicNarration = isObject ? subtopic.narration : '';

                                    if (!subtopicTitle && !subtopicNarration) {
                                      return null;
                                    }

                                    return (
                                      <div
                                        key={subIndex}
                                        className="flex flex-col gap-0.5 bg-primary-500/5 border border-primary-500/20 rounded px-2 py-1"
                                      >
                                        {subtopicTitle && (
                                          <span className="text-xs font-medium text-primary-400 break-words">
                                            {subtopicTitle}
                                          </span>
                                        )}
                                        {subtopicNarration && (
                                          <span className="text-[11px] text-dark-text-muted break-words">
                                            {subtopicNarration}
                                          </span>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              )}
                              {typeof topic === 'object' && (topic.word_count > 0 || topic.char_count > 0) && (
                                <div className="flex items-center gap-3 mt-2 text-xs text-dark-text-muted">
                                  {topic.word_count > 0 && <span>{topic.word_count} words</span>}
                                  {topic.char_count > 0 && <span>{topic.char_count} chars</span>}
                                </div>
                              )}
                              <div className="flex items-center justify-between mt-2">
                                <p className="text-xs text-primary-400/70">
                                  Click checkbox to select • Click here for details
                                </p>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    const topicTitle = typeof topic === 'string' ? topic : topic.title;
                                    const topicSummary = typeof topic === 'object' ? topic.summary : '';
                                    toast.info(`Topic: ${topicTitle}${topicSummary ? `\n${topicSummary}` : ''}`);
                                  }}
                                  className="text-xs text-blue-400 hover:text-blue-300"
                                >
                                  View Details
                                </button>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-32">
                      <div className="text-center space-y-2">
                        <BookOpen className="w-8 h-8 text-dark-text-muted mx-auto" />
                        <p className="text-sm text-dark-text-muted">No topics extracted</p>
                        <p className="text-xs text-dark-text-muted">
                          Topics will appear here after processing
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Topics Panel Footer */}
                <div className="p-3 border-t border-dark-border">
                  <div className="flex items-center justify-between text-xs text-dark-text-muted">
                    <span>
                      {availableTopics[previewMaterial.id]?.length || 0} topics found
                    </span>
                    <button
                      onClick={() => {
                        // Refresh topics for this material
                        const materialIds = [previewMaterial.id];
                        chapterMaterialsAPI.extractTopics(materialIds)
                          .then(response => {
                            if (response.status) {
                              // Handle new topics structure
                              const topicsData = response.data.topics || {};
                              const processedTopics = {};
                              
                              Object.keys(topicsData).forEach(materialId => {
                                const materialData = topicsData[materialId];
                                if (materialData && materialData.topics) {
                                  processedTopics[materialId] = materialData.topics;
                                  console.log(`📄 Material ${materialId} detected language: ${materialData.detected_language}`);
                                } else if (Array.isArray(materialData)) {
                                  processedTopics[materialId] = materialData;
                                }
                              });
                              
                              setAvailableTopics(prev => ({
                                ...prev,
                                ...processedTopics
                              }));
                              toast.success('Topics refreshed!');
                            }
                          })
                          .catch(error => {
                            toast.error('Failed to refresh topics');
                          });
                      }}
                      className="text-primary-400 hover:text-primary-300 transition-colors"
                    >
                      Refresh
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-dark-border">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4 text-sm text-dark-text-muted">
                  <span>Size: {previewMaterial.file_size ? `${(previewMaterial.file_size / (1024 * 1024)).toFixed(2)} MB` : 'Unknown'}</span>
                  {previewMaterial.created_at && (
                    <span>Uploaded: {new Date(previewMaterial.created_at).toLocaleDateString()}</span>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <Button
                    variant="secondary"
                    size="small"
                    onClick={closePreview}
                  >
                    Close
                  </Button>
                  {previewMaterial.file_path && (
                    <Button
                      variant="primary"
                      size="small"
                      icon={FileText}
                      onClick={() => {
                        window.open(getFileUrl(previewMaterial.file_path), '_blank');
                      }}
                    >
                      Open in New Tab
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}

      {/* Topic-Specific Preview Modal */}
      {showTopicPreview && previewMaterial && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="bg-dark-card rounded-lg border border-dark-border w-full max-w-6xl h-[90vh] flex flex-col"
          >
            {/* Header */}
            <div className="p-4 border-b border-dark-border flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-dark-text-primary">
                  Selected Topics Preview
                </h3>
                <p className="text-sm text-dark-text-muted">
                  {previewMaterial.subject} • {previewMaterial.selectedTopicIndices?.length || 0} topics selected
                </p>
              </div>
              <button
                onClick={() => setShowTopicPreview(false)}
                className="p-2 hover:bg-dark-bg/50 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-dark-text-muted" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 p-4 flex gap-4">
              {/* PDF Viewer */}
              <div className="flex-1 bg-dark-bg/60 rounded-lg border border-dark-border overflow-hidden">
                {previewMaterial.file_path ? (
                  <iframe
                    src={getFileUrl(previewMaterial.file_path)}
                    className="w-full h-full"
                    title={`Selected Topics Preview: ${previewMaterial.subject}`}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center space-y-3">
                      <FileText className="w-12 h-12 text-dark-text-muted mx-auto" />
                      <p className="text-dark-text-muted">PDF preview not available</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Selected Topics Panel */}
              <div className="w-80 bg-dark-card rounded-lg border border-dark-border flex flex-col">
                <div className="p-4 border-b border-dark-border">
                  <div className="flex items-center gap-2">
                    <CheckSquare className="w-5 h-5 text-green-400" />
                    <h4 className="font-semibold text-dark-text-primary">Selected Topics</h4>
                  </div>
                  <p className="text-xs text-dark-text-muted mt-1">
                    {previewMaterial.selectedTopicIndices?.length || 0} topics selected for preview
                  </p>
                </div>
                
                <div className="flex-1 p-4 overflow-y-auto">
                  <div className="space-y-3">
                    {previewMaterial.selectedTopicIndices?.map((topicIndex) => {
                      const topic = availableTopics[previewMaterial.id]?.[topicIndex];
                      if (!topic) return null;
                      
                      return (
                        <div
                          key={topicIndex}
                          className="p-3 bg-green-500/10 rounded-lg border border-green-500/20"
                        >
                          <div className="flex items-start gap-3">
                            <div className="w-6 h-6 bg-green-500/20 rounded-full flex items-center justify-center flex-shrink-0">
                              <span className="text-xs font-medium text-green-400">{topicIndex + 1}</span>
                            </div>
                            <div className="flex-1 min-w-0">
                              <h5 className="text-sm font-medium text-dark-text-primary break-words">
                                {typeof topic === 'string' ? topic : topic.title}
                              </h5>
                              {typeof topic === 'object' && topic.summary && (
                                <p className="text-xs text-dark-text-muted mt-1 break-words">
                                  {topic.summary}
                                </p>
                              )}
                              {typeof topic === 'object' && topic.subtopics && topic.subtopics.length > 0 && (
                                <div className="mt-2 space-y-1">
                                  {topic.subtopics.map((subtopic, subIndex) => {
                                    const isObject = subtopic && typeof subtopic === 'object';
                                    const subtopicTitle = isObject ? subtopic.title : subtopic;
                                    const subtopicNarration = isObject ? subtopic.narration : '';

                                    if (!subtopicTitle && !subtopicNarration) {
                                      return null;
                                    }

                                    return (
                                      <div
                                        key={subIndex}
                                        className="flex flex-col gap-0.5 bg-green-500/5 border border-green-500/20 rounded px-2 py-1"
                                      >
                                        {subtopicTitle && (
                                          <span className="text-xs font-medium text-green-400 break-words">
                                            {subtopicTitle}
                                          </span>
                                        )}
                                        {subtopicNarration && (
                                          <span className="text-[11px] text-dark-text-muted break-words">
                                            {subtopicNarration}
                                          </span>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Footer Actions */}
                <div className="p-4 border-t border-dark-border">
                  <div className="flex gap-2">
                    <Button
                      variant="secondary"
                      size="small"
                      onClick={() => setShowTopicPreview(false)}
                      className="flex-1"
                    >
                      Close
                    </Button>
                    <Button
                      variant="primary"
                      size="small"
                      onClick={() => handleGenerateTopicPDF(previewMaterial)}
                      disabled={generatingPDF}
                      className="flex-1"
                    >
                      {generatingPDF ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin mr-2" />
                          Generating...
                        </>
                      ) : (
                        'Generate PDF'
                      )}
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}

      {/* Generated Lecture Modal */}
    </div>
  );
};

export default SelectedMaterials;
