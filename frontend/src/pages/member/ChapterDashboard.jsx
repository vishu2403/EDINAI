/**
* Chapter Dashboard - Dashboard for Chapter Management members
*/

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { dashboardAPI, chapterMaterialsAPI, lectureAPI, getFileUrl } from '../../utils/api';
import Button from '../../components/ui/Button';
import InputField from '../../components/ui/InputField';
import FileUploader from '../../components/ui/FileUploader';
import {
  BookOpen,
  FileText,
  LogOut,
  Bell,
  BookPlus,
  Layers,
  GraduationCap,
  Loader2,
  Search,
  Filter,
  Trash2,
  PlusCircle,
  Home,
  RefreshCw,
  Eye,
  ChevronLeft,
  ChevronRight,
  X,
  CheckSquare,
  Square,
  MessageSquare,
  Sparkles,
} from 'lucide-react';
import toast from 'react-hot-toast';

const PAGE_SIZE = 10;
const PDF_MAX_SIZE_BYTES = 15 * 1024 * 1024;

const FILTER_LABELS = {
  std: 'Class',
  subject: 'Subject',
  sem: 'Semester',
  board: 'Board',
};

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', description: 'Dashboard home', icon: Home },
  { id: 'filters', label: 'Filters', description: 'Adjust material filters', icon: Filter },
  { id: 'add', label: 'Add', description: 'Upload new chapter PDFs', icon: PlusCircle },
];

const UPLOAD_REQUIRED_FIELDS = ['std', 'subject'];
const SEMESTER_OPTIONS = ['Sem 1', 'Sem 2'];

const sanitizeFilters = (rawFilters) => {
  const cleaned = {};
  Object.entries(rawFilters).forEach(([key, value]) => {
    if (value && value.trim()) {
      cleaned[key] = value.trim();
    }
  });
  return cleaned;
};

const formatFileSize = (bytes) => {
  if (!bytes || Number.isNaN(bytes)) {
    return '—';
  }

  const mb = bytes / (1024 * 1024);
  if (mb < 0.5) {
    const kb = bytes / 1024;
    return `${kb.toFixed(1)} KB`;
  }

  return `${mb.toFixed(2)} MB`;
};

const formatDate = (value) => {
  if (!value) return '—';

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return '—';
  }

  return date.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
};

const formatTime = (value) => {
  if (!value) return '—';

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return '—';
  }

  return date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  });
};

const normalizeTopicsByMaterial = (payload) => {
  if (!payload) {
    return {};
  }

  if (Array.isArray(payload)) {
    return payload.reduce((acc, item) => {
      if (!item || typeof item !== 'object') {
        return acc;
      }

      const materialId =
        item.material_id ??
        item.materialId ??
        item.id ??
        item.materialID ??
        null;

      if (materialId == null) {
        return acc;
      }

      const key = String(materialId);
      const normalizedEntry = {
        ...item,
        material_id: materialId,
        topics: Array.isArray(item.topics) ? item.topics : [],
      };

      if (!normalizedEntry.chapter_title && normalizedEntry.chapterTitle) {
        normalizedEntry.chapter_title = normalizedEntry.chapterTitle;
      }

      if (!normalizedEntry.chapter_titles && normalizedEntry.chapterTitles) {
        normalizedEntry.chapter_titles = normalizedEntry.chapterTitles;
      }

      acc[key] = normalizedEntry;
      return acc;
    }, {});
  }

  if (typeof payload === 'object') {
    return payload;
  }

  return {};
};

const ChapterDashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const [dashboardData, setDashboardData] = useState(null);
  const [loadingDashboard, setLoadingDashboard] = useState(true);
  const [materials, setMaterials] = useState([]);
  const [materialsLoading, setMaterialsLoading] = useState(false);
  const [filters, setFilters] = useState({
    std: '',
    subject: '',
    sem: '',
    board: '',
  });
  const [recentUploads, setRecentUploads] = useState([]);
  const [recentTopics, setRecentTopics] = useState([]);
  const [appliedFilters, setAppliedFilters] = useState({
    std: '',
    subject: '',
    sem: '',
    board: '',
  });
  const [subjectOptions, setSubjectOptions] = useState([]);
  const [subjectOptionsLoading, setSubjectOptionsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [uploadForm, setUploadForm] = useState({
    std: '',
    subject: '',
  });
  const [chapterTitle, setChapterTitle] = useState('');
  const [pdfFiles, setPdfFiles] = useState([]);
  const [uploaderResetKey, setUploaderResetKey] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [deleteLoadingId, setDeleteLoadingId] = useState(null);
  const [isFilterPanelOpen, setIsFilterPanelOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [activeSection, setActiveSection] = useState('overview');
  const [suggestedMaterials, setSuggestedMaterials] = useState([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const [suggestionsUnlocked, setSuggestionsUnlocked] = useState(false);
  const [selectedChapterIds, setSelectedChapterIds] = useState([]);
  const [topicGenerationMode, setTopicGenerationMode] = useState(false);
  const [skipContextMode, setSkipContextMode] = useState(false);
  const [selectedSuggestionId, setSelectedSuggestionId] = useState(null);
  const [selectedSuggestionIds, setSelectedSuggestionIds] = useState([]);
  const [topicExtractionLoading, setTopicExtractionLoading] = useState(false);
  const [generatedTopicsByMaterial, setGeneratedTopicsByMaterial] = useState({});
  const [selectedTopicsByMaterial, setSelectedTopicsByMaterial] = useState({});
  const [activeMaterialId, setActiveMaterialId] = useState(null);
  const [showMergedPreview, setShowMergedPreview] = useState(false);
  const [generatedTopicsError, setGeneratedTopicsError] = useState(null);
  const [manualTopicTitle, setManualTopicTitle] = useState('');
  const [manualTopicSummary, setManualTopicSummary] = useState('');
  const [manualTopicSubtopics, setManualTopicSubtopics] = useState('');
  const [savedTopicsByMaterial, setSavedTopicsByMaterial] = useState({});
  const [expandedSavedMaterials, setExpandedSavedMaterials] = useState({});
  const [chapterTitlesByMaterial, setChapterTitlesByMaterial] = useState({});
  const [assistantQuery, setAssistantQuery] = useState('');
  const [assistantTemperature, setAssistantTemperature] = useState(0.2);
  const [assistantPlanLabel, setAssistantPlanLabel] = useState('');
  const [assistantResponses, setAssistantResponses] = useState([]);
  const [assistantSelections, setAssistantSelections] = useState({});
  const [assistantLoading, setAssistantLoading] = useState(false);
  const [assistantSaving, setAssistantSaving] = useState(false);
  const [assistantError, setAssistantError] = useState(null);
  const assistantResponseIdRef = useRef(0);
  const assistantPlanOptions = ['', '20k', '50k', '100k'];
  const [lectureOverview, setLectureOverview] = useState({ items: [], total: 0 });
  const [lectureOverviewLoading, setLectureOverviewLoading] = useState(true);
  const [lectureOverviewError, setLectureOverviewError] = useState(null);
  const latestAssistantMeta = useMemo(
    () => (assistantResponses.length ? assistantResponses[assistantResponses.length - 1]?.meta || null : null),
    [assistantResponses]
  );
  const selectedAssistantSuggestions = useMemo(
    () =>
      assistantResponses.flatMap((response) => {
        const indices = assistantSelections[response.id] || [];
        if (!Array.isArray(indices) || !indices.length) {
          return [];
        }
        return indices
          .map((index) => response.suggestions?.[index])
          .filter(Boolean);
      }),
    [assistantResponses, assistantSelections]
  );
  const assistantTotalSuggestions = useMemo(
    () => assistantResponses.reduce((sum, response) => sum + (response.suggestions?.length || 0), 0),
    [assistantResponses]
  );
  const assistantSelectedCount = selectedAssistantSuggestions.length;
  const assistantHasSelections = assistantSelectedCount > 0;
  const assistantHasSuggestions = assistantTotalSuggestions > 0;
  const hasLectureOverviewItems = lectureOverview.items.length > 0;

  const selectedSuggestion = useMemo(
    () => suggestedMaterials.find((material) => material.id === (activeMaterialId || selectedSuggestionId)) || null,
    [suggestedMaterials, activeMaterialId, selectedSuggestionId]
  );

  const activeMaterialSelections = useMemo(
    () => (activeMaterialId ? selectedTopicsByMaterial[activeMaterialId] || [] : []),
    [activeMaterialId, selectedTopicsByMaterial]
  );

  const hasSelectedTopics = activeMaterialSelections.length > 0;
  const hasAnySelectedTopics = useMemo(
    () => Object.values(selectedTopicsByMaterial).some((entries) => (entries || []).length > 0),
    [selectedTopicsByMaterial]
  );

  const recentTopicsMap = useMemo(() => {
    const map = {};
    recentTopics.forEach((item) => {
      if (item?.material?.id) {
        map[item.material.id] = item.topics || [];
      }
    });
    return map;
  }, [recentTopics]);

  const renderSkipUploadPanel = () => (
    <div className="space-y-4">
      <div className="rounded-xl border border-dark-border bg-dark-card/60 p-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h3 className="text-lg font-semibold">Quick Upload</h3>
            <p className="text-sm text-dark-text-muted">Review the captured context and upload your chapter PDF directly.</p>
          </div>
          <Button variant="ghost" size="small" onClick={() => setSkipContextMode(false)}>
            Change details
          </Button>
        </div>
        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          <div className="rounded-lg border border-dark-border/70 bg-dark-bg/30 p-3">
            <p className="text-xs text-dark-text-muted">Class</p>
            <p className="text-sm font-semibold text-dark-text-primary">{uploadForm.std || '—'}</p>
          </div>
          <div className="rounded-lg border border-dark-border/70 bg-dark-bg/30 p-3">
            <p className="text-xs text-dark-text-muted">Subject</p>
            <p className="text-sm font-semibold text-dark-text-primary">{uploadForm.subject || '—'}</p>
          </div>
          <div className="rounded-lg border border-dark-border/70 bg-dark-bg/30 p-3">
            <p className="text-xs text-dark-text-muted">Chapter</p>
            <p className="text-sm font-semibold text-dark-text-primary">{chapterTitle || '—'}</p>
          </div>
        </div>
      </div>
      {!topicGenerationMode && renderUploadPanel()}
    </div>
  );

  useEffect(() => {
    loadDashboardData();
    loadRecentTopics();
    loadLectureOverview();
  }, []);

  useEffect(() => {
    if (
      topicGenerationMode &&
      selectedSuggestionId &&
      (!generatedTopicsByMaterial[selectedSuggestionId] || generatedTopicsByMaterial[selectedSuggestionId].length === 0) &&
      !topicExtractionLoading
    ) {
      fetchPersistedTopics(selectedSuggestionId);
    }
  }, [topicGenerationMode, selectedSuggestionId]);

  const loadDashboardData = async () => {
    try {
      const response = await dashboardAPI.getChapterDashboard();
      if (response.status) {
        setDashboardData(response.data);
      }
    } catch (error) {
      console.error('Dashboard data error:', error);
      toast.error('Failed to load dashboard data');
    } finally {
      setLoadingDashboard(false);
    }
  };

  const renderLectureCard = (lecture) => {
    const chapterTitle = lecture.chapter_name || lecture.lecture_title || 'Untitled chapter';
    const subjectLabel = lecture.subject || lecture.material_subject || 'Subject';
    const stdLabel = lecture.std || lecture.material_std || 'Class';
    const timeLabel = formatTime(lecture.created_at);

    const tagList = [lecture.sem || lecture.material_sem, lecture.board || lecture.material_board].filter(Boolean);

    return (
      <div key={lecture.id} className="rounded-2xl border border-dark-border bg-dark-card/70 p-5 shadow-lg shadow-black/10">
        <div className="flex flex-col gap-4">
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-1">
              <p className="text-[13px] uppercase tracking-wide text-primary-200">Subject - {subjectLabel}</p>
              <h4 className="text-lg font-semibold text-dark-text-primary">{chapterTitle}</h4>
              <p className="text-sm text-dark-text-muted">Class {stdLabel} {tagList.length > 0 && `• ${tagList.join(' • ')}`}</p>
            </div>
            <div className="rounded-lg bg-dark-bg/60 px-3 py-1 text-xs text-dark-text-muted">{timeLabel}</div>
          </div>

          <div className="space-y-2 text-sm text-dark-text-muted">
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-primary-300" />
              <span>Material #{lecture.material_id}</span>
            </div>
            {lecture.lecture_title && (
              <div className="flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-primary-300" />
                <span className="text-dark-text-primary">{lecture.lecture_title}</span>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between pt-2">
            <div className="flex items-center gap-3 text-xs text-dark-text-muted">
              <span>{lecture.sem || lecture.material_sem || 'Semester —'}</span>
              <span>•</span>
              <span>{lecture.board || lecture.material_board || 'Board —'}</span>
            </div>
            <div className="flex items-center gap-2">
              <a
                href={getFileUrl(lecture.lecture_json_url || lecture.lecture_link)}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 rounded-lg border border-dark-border px-3 py-1.5 text-xs text-dark-text-primary hover:border-primary-500 hover:text-primary-100"
              >
                <Eye className="h-4 w-4" />
                View Lecture
              </a>
              <Button
                variant="ghost"
                size="small"
                icon={Trash2}
                disabled
                title="Lecture deletion coming soon"
              >
                Delete
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderChapterOverviewCard = (chapter, index) => {
    const hasTopics = chapter.topics && chapter.topics.length > 0;
    const hasVideo = chapter.video !== null;
    
    return (
      <div key={`chapter-${index}`} className="rounded-2xl border border-dark-border bg-dark-card/70 p-5 shadow-lg shadow-black/10">
        <div className="flex gap-4">
          {/* Video Thumbnail Section */}
          <div className="flex-shrink-0">
            {hasVideo ? (
              <div className="w-32 h-20 bg-dark-bg rounded-lg flex items-center justify-center relative">
                <div className="text-xs text-dark-text-muted absolute bottom-1 right-1">
                  {formatFileSize(chapter.video.size)}
                </div>
                <GraduationCap className="h-8 w-8 text-dark-text-muted" />
              </div>
            ) : (
              <div className="w-32 h-20 bg-dark-bg rounded-lg flex items-center justify-center">
                <BookOpen className="h-8 w-8 text-dark-text-muted" />
              </div>
            )}
          </div>

          {/* Content Section */}
          <div className="flex-1 space-y-3">
            {/* Header */}
            <div className="space-y-1">
              <p className="text-lg font-semibold text-dark-text-primary">Subject - {chapter.subject}</p>
              <p className="text-sm text-dark-text-muted">Chapter - {chapter.chapter}</p>
            </div>

            {/* Topics List */}
            {hasTopics && (
              <div className="space-y-1">
                {chapter.topics.slice(0, 3).map((topic, topicIndex) => (
                  <div key={topicIndex} className="text-sm text-dark-text-muted">
                    • {topic.title || `Topic ${topicIndex + 1}`}
                  </div>
                ))}
              </div>
            )}

            {/* Size and Actions */}
            <div className="flex items-center justify-between">
              <div className="text-sm text-dark-text-muted">
                Size {chapter.size}
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  className="p-2 text-dark-text-muted hover:text-dark-text-primary transition"
                  onClick={() =>
                    navigate('/chapter-dashboard/edit', {
                      state: { chapter },
                    })
                  }
                >
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                </button>
                <button className="p-2 text-dark-text-muted hover:text-red-400 transition">
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderChapterOverview = () => {
    const chapterOverviewData = dashboardData?.chapter_overview || [];
    const hasChapterData = chapterOverviewData.length > 0;

    return (
      <motion.div className="card p-6 space-y-6" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm uppercase tracking-wide text-dark-text-muted">Chapter Overview</p>
            <h2 className="text-2xl font-bold text-dark-text-primary">Chapter Management</h2>
            <p className="text-sm text-dark-text-muted">
              {hasChapterData ? `${chapterOverviewData.length} chapter${chapterOverviewData.length > 1 ? 's' : ''} available` : 'No chapters uploaded yet'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="small" icon={RefreshCw} onClick={loadDashboardData} loading={loadingDashboard}>
              Refresh
            </Button>
          </div>
        </div>

        {loadingDashboard ? (
          <div className="flex min-h-[160px] items-center justify-center text-dark-text-muted">
            <Loader2 className="h-5 w-5 animate-spin" />
            <span className="ml-2 text-sm">Loading chapter overview…</span>
          </div>
        ) : hasChapterData ? (
          <div className="grid gap-4 md:grid-cols-2">
            {chapterOverviewData.map((chapter, index) => renderChapterOverviewCard(chapter, index))}
          </div>
        ) : (
          <div className="rounded-xl border border-dashed border-dark-border/80 bg-dark-card/40 p-8 text-center">
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full border border-dark-border">
              <BookOpen className="h-5 w-5 text-dark-text-muted" />
            </div>
            <p className="text-sm text-dark-text-muted">No chapters uploaded yet. Start by uploading a chapter PDF to get started.</p>
          </div>
        )}
      </motion.div>
    );
  };

  const loadLectureOverview = async () => {
    setLectureOverviewLoading(true);
    setLectureOverviewError(null);
    try {
      const response = await lectureAPI.getLectureOverview();
      if (response.status) {
        const data = response.data || {};
        const items = Array.isArray(data.items) ? data.items : [];
        setLectureOverview({
          items,
          total: typeof data.total === 'number' ? data.total : items.length,
        });
      } else {
        setLectureOverview({ items: [], total: 0 });
      }
    } catch (error) {
      console.error('Lecture overview load error:', error);
      setLectureOverviewError(error.response?.data?.detail || 'Failed to load lecture overview');
      toast.error('Failed to load lecture overview');
    } finally {
      setLectureOverviewLoading(false);
    }
  };

  const handleToggleSavedTopics = async (materialId) => {
    const isExpanded = expandedSavedMaterials[materialId];
    if (isExpanded) {
      setExpandedSavedMaterials({});
      return;
    }

    const loaded = await fetchPersistedTopics(materialId, { notify: false, updateGenerated: true });
    if (loaded) {
      setSelectedSuggestionId(materialId);
      setTopicGenerationMode(true);
      setExpandedSavedMaterials({ [materialId]: true });
    } else {
      toast.error('No topics stored yet');
    }
  };

  const handleUseSavedTopics = async (materialId) => {
    const loaded = await fetchPersistedTopics(materialId, { notify: true, updateGenerated: true });
    if (loaded) {
      setSelectedSuggestionId(materialId);
      setTopicGenerationMode(true);
      setExpandedSavedMaterials({ [materialId]: true });
    }
  };

  const renderTopicCard = (topic, index, materialId) => {
    const isStringTopic = typeof topic === 'string';
    const safeTopic = topic || {};
    const title = isStringTopic ? topic : safeTopic.title || `Topic ${index + 1}`;
    const summary = !isStringTopic && safeTopic.summary;
    const subtopics = !isStringTopic && Array.isArray(safeTopic.subtopics)
      ? safeTopic.subtopics.filter((sub) => {
        if (!sub) return false;
        if (typeof sub === 'string') {
          return sub.trim().length > 0;
        }
        return Boolean(sub.title?.trim() || sub.narration?.trim());
      })
      : [];
    const materialSelections = selectedTopicsByMaterial[materialId] || [];
    const isSelected = materialSelections.includes(index);
    const SelectionIcon = isSelected ? CheckSquare : Square;

    return (
      <div
        key={`${title}-${index}`}
        className={`rounded-lg border px-4 py-3 transition hover:border-primary-500/60 ${isSelected ? 'border-primary-500 bg-primary-500/5' : 'border-dark-border/60 bg-dark-bg/40'
          }`}
      >
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="space-y-1">
            <p className="text-[11px] uppercase tracking-wide text-dark-text-muted">
              Topic {index + 1}
            </p>
            <p className="text-sm font-semibold text-dark-text-primary">
              {title}
            </p>
            {summary && (
              <p className="text-xs text-dark-text-muted leading-snug">
                {summary}
              </p>
            )}
          </div>
          <div className="flex flex-col items-end gap-2 text-xs text-dark-text-muted">
            {subtopics.length > 0 && (
              <span>{subtopics.length} subtopic{subtopics.length > 1 ? 's' : ''}</span>
            )}
            <button
              type="button"
              onClick={() => toggleGeneratedTopicSelection(materialId, index)}
              className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-medium transition ${isSelected
                  ? 'border-primary-500/60 bg-primary-500/10 text-primary-100'
                  : 'border-dark-border/60 bg-dark-card/40 text-dark-text-muted hover:text-dark-text-primary'
                }`}
            >
              <SelectionIcon className="h-3.5 w-3.5" />
              {isSelected ? 'Selected' : 'Select'}
            </button>
          </div>
        </div>

        {subtopics.length > 0 && (
          <div className="mt-3 space-y-2">
            {subtopics.map((sub, subIndex) => {
              const isObject = typeof sub === 'object';
              const subtopicTitle = isObject ? sub?.title : sub;
              const subtopicNarration = isObject ? sub?.narration : '';
              const subLabel = `${index + 1}.${subIndex + 1}`;

              return (
                <div
                  key={`${title}-sub-${subIndex}`}
                  className="rounded-md border border-dark-border/50 bg-dark-card/40 px-3 py-2"
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-[11px] font-medium text-primary-300">{subLabel}</span>
                    <div className="flex-1 space-y-1">
                      {subtopicTitle && (
                        <p className="text-xs font-semibold text-dark-text-primary">
                          {subtopicTitle}
                        </p>
                      )}
                      {subtopicNarration && (
                        <p className="text-[11px] text-dark-text-muted leading-snug">
                          {subtopicNarration}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  const handleGenerateMergedLecture = () => {
    const selectedMaterialEntries = Object.entries(selectedTopicsByMaterial).filter(
      ([, indices]) => Array.isArray(indices) && indices.length > 0
    );

    if (!selectedMaterialEntries.length) {
      toast.error('Select at least one topic before generating the lecture');
      return;
    }

    const payloads = selectedMaterialEntries.map(([materialId, indices]) => {
      const idNumber = Number(materialId);
      const topics = generatedTopicsByMaterial[idNumber] || [];
      const selectedTopics = indices
        .map((topicIndex) => ({ topicIndex, topic: topics[topicIndex] }))
        .filter(({ topic }) => Boolean(topic));

      return {
        materialId: idNumber,
        indices,
        topics,
        selectedTopics,
      };
    });

    const flattenedTopics = payloads.flatMap((item) =>
      item.selectedTopics.map(({ topic, topicIndex }) => ({
        ...topic,
        __materialId: item.materialId,
        __topicIndex: topicIndex,
      }))
    );

    const flattenedSelectedTopics = flattenedTopics.map((topic) => ({
      title: typeof topic === 'string' ? topic : topic?.title,
      summary: typeof topic === 'object' ? topic?.summary : '',
      subtopics: typeof topic === 'object' && Array.isArray(topic?.subtopics) ? topic.subtopics : [],
    }));

    if (!flattenedTopics.length) {
      toast.error('Unable to prepare merged topics. Please re-select and try again.');
      return;
    }

    const primaryPayload = payloads[0];

    const resolveMaterialById = (materialId) => {
      if (!materialId) return null;
      return (
        materials.find((item) => item.id === materialId) ||
        suggestedMaterials.find((item) => item.id === materialId) ||
        null
      );
    };

    const primaryMaterial = resolveMaterialById(primaryPayload.materialId);
    const fallbackMaterial =
      primaryMaterial ||
      ({
        id: primaryPayload.materialId,
        subject: 'Merged Lecture',
        std: uploadForm.std || appliedFilters.std || '—',
        board: uploadForm.board || appliedFilters.board || '—',
        sem: uploadForm.sem || appliedFilters.sem || '—',
        chapter_number: chapterTitlesByMaterial[primaryPayload.materialId] || chapterTitle || '—',
      });

    const mergedState = {
      material: fallbackMaterial,
      selectedTopicIndices: flattenedTopics.map((_, index) => index),
      topics: flattenedTopics,
      topicsOverride: flattenedTopics,
      selectedTopics: flattenedSelectedTopics,
      sourceMaterialIds: payloads.map((p) => p.materialId),
    };

    localStorage.setItem('inai_pending_lecture_selection', JSON.stringify(mergedState));

    navigate('/generate-lecture', {
      state: mergedState,
    });
  };

  const loadRecentTopics = async (limit = 10) => {
    try {
      const response = await chapterMaterialsAPI.recentTopics(limit);
      if (response.status) {
        setRecentTopics(response.data?.items || []);
      }
    } catch (error) {
      console.error('Recent topics fetch error:', error);
    }
  };

  const fetchPersistedTopics = async (materialId, { notify = false, updateGenerated = true } = {}) => {
    try {
      const response = await chapterMaterialsAPI.getMaterialTopics(materialId);
      if (response.status) {
        const topicsList = response.data?.topics || [];
        const metadata = response.data?.topics_metadata || {};
        const persistedChapterTitle = (metadata.chapter_title || response.data?.material?.chapter_number || '').trim();
        if (persistedChapterTitle) {
          setChapterTitlesByMaterial((prev) => ({
            ...prev,
            [materialId]: persistedChapterTitle,
          }));
        }
        if (topicsList.length) {
          setSavedTopicsByMaterial((prev) => ({
            ...prev,
            [materialId]: topicsList,
          }));
          if (updateGenerated) {
            setGeneratedTopicsByMaterial((prev) => ({
              ...prev,
              [materialId]: topicsList,
            }));
            setSelectedTopicsByMaterial((prev) => ({
              ...prev,
              [materialId]: [],
            }));
            setActiveMaterialId(materialId);
            resetManualTopicForm();
            if (notify) {
              toast.success('Loaded saved topics');
            }
          }
          return true;
        }
      }
    } catch (error) {
      console.error('Fetch persisted topics error:', error);
    }
    return false;
  };

  const handleManualTopicAdd = () => {
    if (!manualTopicTitle.trim()) {
      toast.error('Enter a topic title before adding');
      return;
    }

    const formattedSubtopics = manualTopicSubtopics
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => ({ title: line, narration: '' }));

    const newTopic = {
      title: manualTopicTitle.trim(),
      summary: manualTopicSummary.trim(),
      subtopics: formattedSubtopics,
      isManual: true,
    };

    if (!activeMaterialId) {
      toast.error('Select a chapter panel to add topics');
      return;
    }

    setGeneratedTopicsByMaterial((prev) => {
      const currentTopics = prev[activeMaterialId] || [];
      const nextTopics = [...currentTopics, newTopic];
      return {
        ...prev,
        [activeMaterialId]: nextTopics,
      };
    });

    setSelectedTopicsByMaterial((prev) => {
      const currentSelection = prev[activeMaterialId] || [];
      const newIndex = (generatedTopicsByMaterial[activeMaterialId] || []).length;
      return {
        ...prev,
        [activeMaterialId]: currentSelection.includes(newIndex)
          ? currentSelection
          : [...currentSelection, newIndex],
      };
    });

    resetManualTopicForm();
    toast.success('Topic added to the list');
  };

  const loadMaterials = async (filtersOverride = appliedFilters) => {
    const params = sanitizeFilters(filtersOverride);
    const hasRequiredFilters = UPLOAD_REQUIRED_FIELDS.every((key) => params[key]);
    if (!hasRequiredFilters) {
      setMaterials([]);
      return;
    }

    setMaterialsLoading(true);

    try {
      const response = await chapterMaterialsAPI.listMaterials(params);
      if (response.status) {
        setMaterials(response.data?.materials || []);
      }
    } catch (error) {
      console.error('Chapter materials fetch error:', error);
      const message = error.response?.data?.detail || 'Failed to load chapter materials';
      toast.error(message);
    } finally {
      setMaterialsLoading(false);
    }
  };

  const filteredMaterials = useMemo(() => {
    if (!searchQuery.trim()) {
      return materials;
    }

    const query = searchQuery.trim().toLowerCase();
    return materials.filter((material) => {
      const searchable = [
        material.subject,
        material.std,
        material.sem,
        material.board,
        material.chapter_number,
        material.file_name,
      ]
        .filter(Boolean)
        .map((value) => String(value).toLowerCase());

      return searchable.some((value) => value.includes(query));
    });
  }, [materials, searchQuery]);

  const filteredCount = filteredMaterials.length;
  const totalPages = filteredCount === 0 ? 1 : Math.ceil(filteredCount / PAGE_SIZE);

  const paginatedMaterials = useMemo(() => {
    const start = (currentPage - 1) * PAGE_SIZE;
    return filteredMaterials.slice(start, start + PAGE_SIZE);
  }, [filteredMaterials, currentPage]);

  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages);
    }
  }, [totalPages, currentPage]);

  const hasActiveFilters = useMemo(
    () => Object.values(appliedFilters).some((value) => Boolean(value)),
    [appliedFilters]
  );

  useEffect(() => {
    if (!topicGenerationMode || !selectedSuggestionId) {
      return;
    }

    handleGenerateTopics(selectedSuggestionId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topicGenerationMode, selectedSuggestionId]);

  const activeFilterChips = useMemo(
    () =>
      Object.entries(appliedFilters)
        .filter(([, value]) => Boolean(value))
        .map(([key, value]) => ({ key, value })),
    [appliedFilters]
  );

  const hasCompleteAppliedFilters = useMemo(
    () => UPLOAD_REQUIRED_FIELDS.every((key) => Boolean(appliedFilters[key] && appliedFilters[key].trim())),
    [appliedFilters]
  );

  const hasCompleteUploadContext = useMemo(
    () => UPLOAD_REQUIRED_FIELDS.every((key) => uploadForm[key] && uploadForm[key].trim()),
    [uploadForm]
  );

  const hasSuggestionContext = useMemo(
    () => Boolean(uploadForm.std.trim() && uploadForm.subject.trim() && chapterTitle.trim()),
    [uploadForm.std, uploadForm.subject, chapterTitle]
  );

  useEffect(() => {
    if (hasCompleteAppliedFilters) {
      loadMaterials(appliedFilters);
    } else {
      setMaterials([]);
      setCurrentPage(1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appliedFilters, hasCompleteAppliedFilters]);

  const handleFilterChange = (field) => (event) => {
    const value = event.target.value;
    setFilters((prev) => {
      const updated = {
        ...prev,
        [field]: value,
      };
      if (field === 'std') {
        updated.subject = '';
      }
      return updated;
    });
  };

  const handleApplyFilters = () => {
    setAppliedFilters({
      std: filters.std.trim(),
      subject: filters.subject.trim(),
      sem: filters.sem.trim(),
      board: filters.board.trim(),
    });
    setCurrentPage(1);
    if (!isFilterPanelOpen) {
      setIsFilterPanelOpen(true);
    }
    setActiveSection('filters');
  };

  const handleClearFilters = () => {
    const cleared = { std: '', subject: '', sem: '', board: '' };
    setFilters(cleared);
    setAppliedFilters(cleared);
    setCurrentPage(1);
  };

  const handleSearchChange = (event) => {
    setSearchQuery(event.target.value);
    setCurrentPage(1);
  };

  const handleRefresh = () => {
    if (!hasCompleteAppliedFilters) {
      toast.error('Apply class, subject, semester, and board filters before refreshing');
      return;
    }
    loadMaterials(appliedFilters);
  };

  useEffect(() => {
    const stdValue = filters.std.trim();

    if (!stdValue) {
      setSubjectOptions([]);
      setSubjectOptionsLoading(false);
      setFilters((prev) => (prev.subject ? { ...prev, subject: '' } : prev));
      return;
    }

    let cancelled = false;

    const loadSubjectsForClass = async () => {
      setSubjectOptionsLoading(true);
      try {
        const response = await chapterMaterialsAPI.listMaterials({ std: stdValue });
        if (!response.status || cancelled) return;

        const materialsForClass = response.data?.materials || [];
        const uniqueSubjects = Array.from(
          new Set(
            materialsForClass
              .map((material) => material.subject)
              .filter((subject) => Boolean(subject && subject.trim()))
              .map((subject) => subject.trim())
          )
        );

        setSubjectOptions(uniqueSubjects);

        if (uniqueSubjects.length === 0) {
          setFilters((prev) => (prev.subject ? { ...prev, subject: '' } : prev));
        } else {
          setFilters((prev) => {
            if (prev.subject && !uniqueSubjects.includes(prev.subject)) {
              return { ...prev, subject: '' };
            }
            return prev;
          });
        }
      } catch (error) {
        if (!cancelled) {
          console.error('Failed to load subjects for class:', error);
          toast.error('Unable to load subjects for the selected class');
          setSubjectOptions([]);
        }
      } finally {
        if (!cancelled) {
          setSubjectOptionsLoading(false);
        }
      }
    };

    loadSubjectsForClass();

    return () => {
      cancelled = true;
    };
  }, [filters.std]);

  useEffect(() => {
    if (activeSection !== 'add') return;

    setUploadForm((prev) => {
      let updated = false;
      const next = { ...prev };

      UPLOAD_REQUIRED_FIELDS.forEach((key) => {
        const incoming = appliedFilters[key];
        if (incoming && (!prev[key] || prev[key] === '')) {
          next[key] = incoming;
          updated = true;
        }
      });
      return updated ? next : prev;
    });
  }, [appliedFilters, activeSection]);

  useEffect(() => {
    if (activeSection !== 'add' || !suggestionsUnlocked) {
      setSuggestedMaterials([]);
      setSuggestionsLoading(false);
      return;
    }

    const stdValue = uploadForm.std.trim();
    const subjectValue = uploadForm.subject.trim();
    if (!stdValue || !subjectValue) {
      setSuggestedMaterials([]);
      setSuggestionsLoading(false);
      return;
    }

    const subjectQuery = subjectValue.toLowerCase();

    let cancelled = false;

    const fetchSuggestions = async () => {
      setSuggestionsLoading(true);
      try {
        // Pass both std and subject to backend for proper filtering
        const params = { std: stdValue };
        if (subjectQuery) {
          params.subject = subjectQuery;
        }
        
        const response = await chapterMaterialsAPI.listMaterials(params);
        if (!cancelled && response.status) {
          const materials = response.data?.materials || [];
          // Backend handles filtering, no need for client-side filtering
          setSuggestedMaterials(materials);
          
          // Auto-select all suggestion IDs for selection API
          const suggestionIds = materials.map(material => material.id);
          setSelectedChapterIds(suggestionIds);
        }
      } catch (error) {
        if (!cancelled) {
          console.error('Failed to fetch chapter suggestions:', error);
          toast.error('Unable to fetch chapter suggestions right now');
          setSuggestedMaterials([]);
        }
      } finally {
        if (!cancelled) {
          setSuggestionsLoading(false);
        }
      }
    };

    const timeoutId = setTimeout(fetchSuggestions, 350);

    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
    };
  }, [uploadForm.std, uploadForm.subject, activeSection, suggestionsUnlocked]);

  const handleUploadFieldChange = (field) => (event) => {
    const value = event.target.value;
    setUploadForm((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSelectChapters = async () => {
    try {
      if (!selectedChapterIds.length) {
        toast.error('Please select at least one chapter');
        return;
      }

      const response = await chapterMaterialsAPI.selectMultipleChapters({
        selected_ids: selectedChapterIds
      });

      if (response.status) {
        toast.success(`Successfully selected ${response.data.total_chapters} chapters`);
        console.log('Selected chapters count:', response.data.total_chapters);
      }
    } catch (error) {
      console.error('Failed to select chapters:', error);
      toast.error('Failed to select chapters');
    }
  };

  const handleUpload = async () => {
    try {
      if (!hasCompleteUploadContext) {
        toast.error('Please provide class, subject, semester, and board for this upload');
        return;
      }
      if (!chapterTitle?.trim()) {
        toast.error('Please enter chapter name');
        return;
      }
      if (!pdfFiles.length) {
        toast.error('Please select a PDF to upload');
        return;
      }

      const file = pdfFiles[0];
      const formData = new FormData();
      formData.append('std', uploadForm.std.trim());
      formData.append('sem', uploadForm.sem.trim());
      formData.append('board', uploadForm.board.trim());
      formData.append('subject', uploadForm.subject.trim());
      formData.append('chapter_number', chapterTitle.trim());
      formData.append('pdf_file', file);

      setUploading(true);
      const response = await chapterMaterialsAPI.uploadMaterial(formData);
      if (response.status) {
        toast.success('PDF uploaded');
        const uploadedMaterial = response.data?.material;
        if (uploadedMaterial) {
          setRecentUploads((prev) => [uploadedMaterial, ...prev].slice(0, 5));
          setSuggestedMaterials((prev) => {
            const exists = prev.some((material) => material.id === uploadedMaterial.id);
            if (exists) {
              return prev.map((material) => (material.id === uploadedMaterial.id ? uploadedMaterial : material));
            }
            return [uploadedMaterial, ...prev];
          });
        }
        loadRecentTopics();
        setChapterTitle('');
        setPdfFiles([]);
        setUploaderResetKey((prev) => prev + 1);
        loadMaterials(appliedFilters);
        if (activeSection === 'add') {
          resetUploadForm();
        } else {
          setActiveSection('overview');
        }

        if (uploadedMaterial) {
          setSuggestionsUnlocked(true);
          setSelectedSuggestionId(uploadedMaterial.id);
          setTopicGenerationMode(true);
          handleGenerateTopics(uploadedMaterial.id, {
            forceExtract: true,
          });
        }
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Upload failed';
      toast.error(message);
    } finally {
      setUploading(false);
    }
  };

  const resetUploadForm = () => {
    setUploadForm({
      std: appliedFilters.std || '',
      subject: appliedFilters.subject || '',
      sem: appliedFilters.sem || '',
      board: appliedFilters.board || '',
    });
    setChapterTitle('');
    setPdfFiles([]);
    setSuggestionsUnlocked(false);
    setTopicGenerationMode(false);
    setSelectedSuggestionId(null);
    setActiveMaterialId(null);
    setTopicExtractionLoading(false);
    setGeneratedTopicsByMaterial({});
    setSelectedTopicsByMaterial({});
    setGeneratedTopicsError(null);
    setSkipContextMode(false);
    setChapterTitlesByMaterial({});
    resetAssistantState();
  };

  const handleAssistantToggleSelection = (responseId, index) => {
    setAssistantSelections((prev) => {
      const current = prev[responseId] || [];
      const exists = current.includes(index);
      const nextSelection = exists
        ? current.filter((value) => value !== index)
        : [...current, index];

      if (nextSelection.length === 0) {
        const { [responseId]: _removed, ...rest } = prev;
        return rest;
      }

      return {
        ...prev,
        [responseId]: nextSelection,
      };
    });
  };

  const handleAssistantSelectAll = (responseId) => {
    setAssistantSelections((prev) => {
      const response = assistantResponses.find((item) => item.id === responseId);
      const total = response?.suggestions?.length || 0;
      if (!total) {
        const { [responseId]: _removed, ...rest } = prev;
        return rest;
      }

      const current = prev[responseId] || [];
      const nextSelection = current.length === total ? [] : Array.from({ length: total }, (_, idx) => idx);

      if (nextSelection.length === 0) {
        const { [responseId]: _cleared, ...rest } = prev;
        return rest;
      }

      return {
        ...prev,
        [responseId]: nextSelection,
      };
    });
  };

  const handleAssistantAsk = async () => {
    if (!activeMaterialId) {
      toast.error('Select or activate a chapter before asking the assistant');
      return;
    }

    if (!assistantQuery.trim()) {
      toast.error('Enter a prompt for the assistant');
      return;
    }

    const payload = {
      user_query: assistantQuery.trim(),
      temperature: Number(assistantTemperature) || 0.2,
      plan_label: assistantPlanLabel?.trim() || undefined,
    };

    setAssistantLoading(true);
    setAssistantError(null);

    try {
      const response = await chapterMaterialsAPI.assistantSuggestTopics(activeMaterialId, payload);
      if (!response?.status) {
        throw new Error(response?.message || 'Failed to get assistant suggestions');
      }

      const data = response?.data || {};
      assistantResponseIdRef.current += 1;
      const responseId = assistantResponseIdRef.current;
      const suggestions = Array.isArray(data.suggestions) ? data.suggestions : [];
      const meta = {
        reply: data.reply,
        planLimit: data.plan_limit,
        maxSuggestions: data.max_suggestions,
        languageLabel: data.language_label,
        existingTopicsCount: data.existing_topics_count,
      };

      const newResponse = {
        id: responseId,
        prompt: assistantQuery.trim(),
        createdAt: new Date().toISOString(),
        suggestions,
        meta,
      };

      setAssistantResponses((prev) => [...prev, newResponse]);
      setAssistantSelections((prev) => ({
        ...prev,
        [responseId]: [],
      }));
      setAssistantQuery('');
      toast.success('Assistant suggestions ready');
    } catch (error) {
      console.error('Assistant suggest error:', error);
      const message = error.response?.data?.detail || error.message || 'Assistant request failed';
      setAssistantError(message);
      toast.error(message);
    } finally {
      setAssistantLoading(false);
    }
  };

  const handleAssistantAddSelected = async () => {
    if (!activeMaterialId) {
      toast.error('Select a chapter panel first');
      return;
    }

    if (!assistantHasSelections) {
      toast.error('Select at least one assistant suggestion to add');
      return;
    }

    const payload = {
      selected_suggestions: selectedAssistantSuggestions,
    };

    setAssistantSaving(true);

    try {
      const response = await chapterMaterialsAPI.assistantAddTopics(activeMaterialId, payload);
      if (!response?.status) {
        throw new Error(response?.message || 'Failed to add assistant topics');
      }

      const data = response?.data || {};
      const addedTopics = data.added_topics || [];
      const skipped = data.skipped_duplicates || [];

      if (addedTopics.length) {
        setGeneratedTopicsByMaterial((prev) => {
          const current = prev[activeMaterialId] || [];
          return {
            ...prev,
            [activeMaterialId]: [...current, ...addedTopics],
          };
        });

        setSavedTopicsByMaterial((prev) => {
          const current = prev[activeMaterialId] || [];
          return {
            ...prev,
            [activeMaterialId]: [...current, ...addedTopics],
          };
        });

        toast.success(`Added ${addedTopics.length} topic${addedTopics.length > 1 ? 's' : ''} from assistant`);
      } else {
        toast('No new topics were added');
      }

      if (skipped.length) {
        toast(`Skipped ${skipped.length} duplicates`, {
          icon: '⚠️',
        });
      }

      setAssistantSelections({});
    } catch (error) {
      console.error('Assistant add topics error:', error);
      const message = error.response?.data?.detail || error.message || 'Failed to add assistant topics';
      toast.error(message);
    } finally {
      setAssistantSaving(false);
    }
  };

  const handleDeleteMaterial = async (materialId) => {
    try {
      setDeleteLoadingId(materialId);
      const response = await chapterMaterialsAPI.deleteMaterial(materialId);
      if (response.status) {
        toast.success('Material deleted');
        loadMaterials(appliedFilters);
      }
    } catch (error) {
      console.error('Delete material error:', error);
      const message = error.response?.data?.detail || 'Failed to delete material';
      toast.error(message);
    } finally {
      setDeleteLoadingId(null);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleToggleFilterPanel = () => {
    setIsFilterPanelOpen((prev) => {
      const next = !prev;
      if (next) {
        setActiveSection('filters');
      }
      return next;
    });
  };

  const handleNavClick = (sectionId) => {
    setActiveSection(sectionId);
    if (sectionId === 'filters') {
      setIsFilterPanelOpen(true);
    } else if (sectionId === 'add') {
      setIsFilterPanelOpen(false);
      setUploadForm({
        std: appliedFilters.std || '',
        subject: appliedFilters.subject || '',
        sem: appliedFilters.sem || '',
        board: appliedFilters.board || '',
      });
      setSuggestionsUnlocked(false);
      setTopicGenerationMode(false);
      setSelectedSuggestionId(null);
      setActiveMaterialId(null);
      setTopicExtractionLoading(false);
      setGeneratedTopicsByMaterial({});
      setSelectedTopicsByMaterial({});
      setGeneratedTopicsError(null);
      setSkipContextMode(false);
      setChapterTitlesByMaterial({});
    } else {
      setIsFilterPanelOpen(false);
    }
  };

  const handleRemoveFilter = (key) => {
    setFilters((prev) => ({
      ...prev,
      [key]: '',
    }));
    setAppliedFilters((prev) => ({
      ...prev,
      [key]: '',
    }));
    setCurrentPage(1);
  };

  const handlePageChange = (direction) => {
    if (direction === 'prev' && currentPage > 1) {
      setCurrentPage((prev) => prev - 1);
    }
    if (direction === 'next' && currentPage < totalPages) {
      setCurrentPage((prev) => prev + 1);
    }
  };

  const handleSaveUploadContext = () => {
    if (!hasSuggestionContext) {
      toast.error('Enter class, subject, and chapter name before saving');
      return;
    }

    setSuggestionsUnlocked(true);
    toast.success('Context saved. Suggestions updated.');
  };

  const handleSkipContext = () => {
    if (!hasSuggestionContext) {
      toast.error('Enter class, subject, and chapter name before skipping');
      return;
    }

    setSuggestionsUnlocked(true);
    setSkipContextMode(true);
    setTopicGenerationMode(false);
    toast.success('Skipping suggestions. Upload your chapter PDF directly.');
  };

  const handleNextClick = () => {
    if (topicGenerationMode) {
      handleProceedToLectureGeneration();
      return;
    }

    if (!hasSuggestionContext) {
      toast.error('Enter class, subject, and chapter name before continuing');
      return;
    }

    if (!suggestionsUnlocked) {
      setSuggestionsUnlocked(true);
      toast.success('Context saved. Suggestions updated.');
    }

    setTopicGenerationMode(true);
  };

  const handleToggleSuggestionSelection = (materialId) => {
    setSelectedSuggestionIds((prev) =>
      prev.includes(materialId) ? prev.filter((id) => id !== materialId) : [...prev, materialId]
    );
  };

  const handleProceedWithSelectedChapters = () => {
    if (!selectedSuggestionIds.length) {
      toast.error('Select at least one chapter before continuing');
      return;
    }

    const firstSelectedId = selectedSuggestionIds[0];
    setSelectedSuggestionId(firstSelectedId);
    setTopicGenerationMode(true);
    handleGenerateTopics(selectedSuggestionIds);
  };

  const resetAssistantState = () => {
    setAssistantQuery('');
    setAssistantTemperature(0.2);
    setAssistantPlanLabel('');
    setAssistantResponses([]);
    setAssistantSelections({});
    setAssistantLoading(false);
    setAssistantSaving(false);
    setAssistantError(null);
    assistantResponseIdRef.current = 0;
  };

  const handleExitTopicMode = () => {
    setTopicGenerationMode(false);
    resetManualTopicForm();
    setExpandedSavedMaterials({});
    setActiveMaterialId(null);
    setShowMergedPreview(false);
    resetAssistantState();
  };

  const handleProceedToLectureGeneration = () => {
    if (!topicGenerationMode) {
      return;
    }

    if (!activeMaterialId) {
      toast.error('Select a chapter suggestion before continuing');
      return;
    }

    const selectedIndices = selectedTopicsByMaterial[activeMaterialId] || [];
    if (selectedIndices.length === 0) {
      toast.error('Select at least one topic before continuing');
      return;
    }

    const topicsForMaterial = generatedTopicsByMaterial[activeMaterialId] || [];
    const selectedTopics = selectedIndices.map((index) => topicsForMaterial[index]);

    setShowMergedPreview(true);
    setTimeout(() => {
      const mergedPanel = document.getElementById('merged-topics-panel');
      if (mergedPanel) {
        mergedPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 150);

    toast.success('Merged topics ready! Showing merged view.');
  };

  // Function to handle generating lecture for selected topics
  const handleGenerateForSelectedTopics = (material, topics, selectedIndices) => {
    if (!material || !topics || selectedIndices.length === 0) {
      toast.error('Invalid selection for lecture generation');
      return;
    }

    const selectedTopics = selectedIndices.map((index) => topics[index]);

    // Show loading state
    toast.loading('Preparing lecture generation...');

    const payload = {
      material,
      selectedTopicIndices: selectedIndices,
      topics,
      selectedTopics,
    };

    localStorage.setItem('inai_pending_lecture_selection', JSON.stringify(payload));

    // Small delay to allow toast to show
    setTimeout(() => {
      navigate('/generate-lecture', {
        state: payload,
      });
      toast.dismiss();
    }, 500);
  };

  const resetManualTopicForm = () => {
    setManualTopicTitle('');
    setManualTopicSummary('');
    setManualTopicSubtopics('');
  };

  const handleGenerateTopics = async (
    materialIdParam,
    { forceExtract = false } = {}
  ) => {
    // Handle both single material and multiple materials (from selectedSuggestionIds)
    let materialIds = [];
    if (Array.isArray(materialIdParam)) {
      materialIds = materialIdParam;
    } else if (materialIdParam) {
      materialIds = [materialIdParam];
    } else if (selectedSuggestionIds.length > 0) {
      materialIds = selectedSuggestionIds;
    } else if (selectedSuggestionId) {
      materialIds = [selectedSuggestionId];
    }

    if (materialIds.length === 0 || topicExtractionLoading) {
      toast.error('Select at least one chapter to extract topics');
      return;
    }

    console.log('🔍 Extracting topics for material IDs:', materialIds);

    setTopicExtractionLoading(true);
    setGeneratedTopicsError(null);
    resetManualTopicForm();

    const toastId = 'topic-extraction';
    const chapterText = materialIds.length === 1 ? 'chapter' : `${materialIds.length} chapters`;
    toast.loading(`Extracting topics from ${chapterText}...`, { id: toastId });

    try {
      const response = await chapterMaterialsAPI.extractTopics(materialIds);
      console.log('📦 API Response:', response);
      toast.dismiss(toastId);

      if (response.status) {
        const topicsMap = normalizeTopicsByMaterial(response.data?.topics);
        console.log('🗺️ Topics Map:', topicsMap);

        // Store topics for each material
        const newSavedTopics = {};
        const newChapterTitles = {};
        let totalTopicsCount = 0;

        materialIds.forEach(matId => {
          const normalizedKey = String(matId);
          const matchingKey = Object.prototype.hasOwnProperty.call(topicsMap, normalizedKey)
            ? normalizedKey
            : matId;

          const mapEntry = topicsMap[matchingKey];
          const entry = Array.isArray(mapEntry)
            ? { topics: mapEntry }
            : (mapEntry && typeof mapEntry === 'object')
              ? mapEntry
              : { topics: [] };

          const rawTopics = entry.topics;
          const topicsList = Array.isArray(rawTopics) ? rawTopics : [];
          const materialIdNumber = Number(matId);
          const materialRecord =
            suggestedMaterials.find((item) => item.id === materialIdNumber) ||
            materials.find((item) => item.id === materialIdNumber) ||
            null;

          const detectedChapterTitle = (
            entry.chapter_title ||
            (Array.isArray(entry.chapter_titles) ? entry.chapter_titles[0] : '') ||
            ''
          ).trim();
          const fallbackChapterTitle = (materialRecord?.chapter_number || '').trim();
          const chapterTitleValue = detectedChapterTitle || fallbackChapterTitle;

          console.log(`📚 Material ${matId}: ${topicsList.length} topics`, topicsList);

          newSavedTopics[materialIdNumber] = topicsList;
          totalTopicsCount += topicsList.length;

          if (chapterTitleValue) {
            newChapterTitles[materialIdNumber] = chapterTitleValue;
          }
        });

        console.log('💾 Saving topics by material:', newSavedTopics);

        setSavedTopicsByMaterial((prev) => {
          const updated = {
            ...prev,
            ...newSavedTopics,
          };
          console.log('📝 Updated savedTopicsByMaterial:', updated);
          return updated;
        });

        setGeneratedTopicsByMaterial((prev) => ({
          ...prev,
          ...newSavedTopics,
        }));

        if (Object.keys(newChapterTitles).length > 0) {
          setChapterTitlesByMaterial((prev) => ({
            ...prev,
            ...newChapterTitles,
          }));
        }

        setSelectedTopicsByMaterial((prev) => {
          const updatedSelections = { ...prev };
          materialIds.forEach((matId) => {
            updatedSelections[matId] = [];
          });
          return updatedSelections;
        });

        const primaryMaterialId = materialIds[0];
        setActiveMaterialId(primaryMaterialId);

        // Expand all materials that have topics
        const expandedMats = {};
        materialIds.forEach(matId => {
          if (newSavedTopics[matId]?.length > 0) {
            expandedMats[matId] = true;
          }
        });
        console.log('📂 Expanded materials:', expandedMats);
        setExpandedSavedMaterials(expandedMats);

        resetManualTopicForm();
        setSkipContextMode(false);

        if (totalTopicsCount === 0) {
          toast.error('No topics were extracted from the selected chapters');
        } else {
          toast.success(`Extracted ${totalTopicsCount} topic${totalTopicsCount > 1 ? 's' : ''} from ${materialIds.length} chapter${materialIds.length > 1 ? 's' : ''}`);
        }
      } else {
        throw new Error(response.message || 'Failed to extract topics');
      }
    } catch (error) {
      console.error('❌ Topic extraction error:', error);
      toast.dismiss(toastId);
      const message = error.response?.data?.detail || error.message || 'Failed to extract topics';
      setGeneratedTopicsError(message);
      toast.error(message);
    } finally {
      setTopicExtractionLoading(false);
    }
  };

  const toggleGeneratedTopicSelection = (materialId, topicIndex) => {
    setSelectedTopicsByMaterial((prev) => {
      const current = prev[materialId] || [];
      if (current.includes(topicIndex)) {
        return {
          ...prev,
          [materialId]: current.filter((index) => index !== topicIndex),
        };
      }
      return {
        ...prev,
        [materialId]: [...current, topicIndex],
      };
    });
    setActiveMaterialId(materialId);
  };

  const handleSelectAllGeneratedTopics = (materialId) => {
    const topics = generatedTopicsByMaterial[materialId] || [];
    setSelectedTopicsByMaterial((prev) => {
      const current = prev[materialId] || [];
      if (topics.length === 0) {
        return {
          ...prev,
          [materialId]: [],
        };
      }
      if (current.length === topics.length) {
        return {
          ...prev,
          [materialId]: [],
        };
      }
      return {
        ...prev,
        [materialId]: topics.map((_, index) => index),
      };
    });
    setActiveMaterialId(materialId);
  };

  

  const renderTopicsForMaterial = (materialId) => {
    const topics = generatedTopicsByMaterial[materialId] || [];
    const material = suggestedMaterials.find((item) => item.id === materialId) ||
      materials.find((item) => item.id === materialId) || null;
    const detectedChapterTitle = chapterTitlesByMaterial[materialId];
    const fallbackChapterLabel = material?.chapter_number || `Chapter ${materialId}`;
    const chapterLabel = detectedChapterTitle || fallbackChapterLabel;
    const showDetectedBadge = Boolean(
      detectedChapterTitle && detectedChapterTitle !== fallbackChapterLabel,
    );
    const selection = selectedTopicsByMaterial[materialId] || [];
    const chapterTitleLabel = chapterLabel || 'Not detected yet';

    return (
      <div key={materialId} className="rounded-xl border border-dark-border/60 bg-dark-card/50 p-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex flex-col">
              <span className="text-[11px] uppercase tracking-wide text-dark-text-muted">Chapter title</span>
              <p className="text-sm font-semibold text-dark-text-primary">{chapterTitleLabel}</p>
            </div>
            {showDetectedBadge ? (
              <p className="text-[11px] text-primary-200">Detected directly from PDF</p>
            ) : (
              <p className="text-[11px] text-dark-text-muted">Using saved chapter name</p>
            )}
            <p className="text-xs text-dark-text-muted">
              Class {material?.std || '—'} · {material?.subject || '—'} · {material?.sem || '—'} · {material?.board || '—'}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-dark-text-muted">{selection.length} selected</span>
            <Button
              variant="secondary"
              size="extraSmall"
              onClick={() => handleSelectAllGeneratedTopics(materialId)}
              disabled={topics.length === 0}
            >
              {selection.length === topics.length && topics.length > 0 ? 'Clear selection' : 'Select all'}
            </Button>
            <Button
              variant={activeMaterialId === materialId ? 'primary' : 'ghost'}
              size="extraSmall"
              onClick={() => setActiveMaterialId(materialId)}
            >
              {activeMaterialId === materialId ? 'Active' : 'Make Active'}
            </Button>
          </div>
        </div>

        <div className="mt-3 space-y-3">
          {topics.length > 0 ? (
            topics.map((topic, index) => renderTopicCard(topic, index, materialId))
          ) : (
            <p className="text-xs text-dark-text-muted">No topics extracted for this chapter yet.</p>
          )}
        </div>
      </div>
    );
  };

  const mergedTopicsPreview = () => {
    if (!topicGenerationMode || !showMergedPreview) {
      return null;
    }

    const selectedMaterialIds = Object.entries(selectedTopicsByMaterial)
      .filter(([, indices]) => Array.isArray(indices) && indices.length > 0)
      .map(([materialId]) => Number(materialId));

    if (selectedMaterialIds.length === 0) {
      return null;
    }

    const totalTopicsSelected = selectedMaterialIds.reduce((count, materialId) => {
      const indices = selectedTopicsByMaterial[materialId] || [];
      return count + indices.length;
    }, 0);

    return (
      <div id="merged-topics-panel" className="rounded-xl border border-primary-500/40 bg-primary-600/10 p-6 space-y-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm font-semibold text-primary-100">Merged Topics Preview</p>
            <p className="text-xs text-primary-200">
              Showing {totalTopicsSelected} topic{totalTopicsSelected > 1 ? 's' : ''} from {selectedMaterialIds.length} chapter{selectedMaterialIds.length > 1 ? 's' : ''}
            </p>
          </div>
          <div className="text-xs text-primary-100">
            Click a chapter below to review selected topics before generating the lecture.
          </div>
        </div>

        <div className="space-y-4">
          {selectedMaterialIds.map((materialId) => {
            const material = suggestedMaterials.find((item) => item.id === materialId) ||
              materials.find((item) => item.id === materialId) || null;
            const topicsForMaterial = generatedTopicsByMaterial[materialId] || [];
            const selectedIndices = selectedTopicsByMaterial[materialId] || [];
            const selectedTopics = selectedIndices.map((index) => ({
              ...topicsForMaterial[index],
              topicNumber: index + 1,
            }));
            const detectedChapterTitle = chapterTitlesByMaterial[materialId];
            const fallbackChapterLabel = material?.chapter_number || `Chapter ${materialId}`;
            const chapterLabel = detectedChapterTitle || fallbackChapterLabel;
            const chapterTitleLabel = chapterLabel || 'Not detected yet';

            return (
              <div key={`preview-${materialId}`} className="rounded-lg border border-primary-500/30 bg-dark-card/80 p-4 space-y-3">
                <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="text-sm font-semibold text-dark-text-primary">Chapter title: {chapterTitleLabel}</p>
                    <p className="text-[11px] text-primary-200">
                      Class {material?.std || '—'} · {material?.subject || '—'} · {material?.sem || '—'} · {material?.board || '—'}
                    </p>
                  </div>
                  <span className="text-xs text-primary-200">{selectedTopics.length} topic{selectedTopics.length > 1 ? 's' : ''} selected</span>
                </div>

                <div className="space-y-2">
                  {selectedTopics.map((topic, idx) => (
                    <div key={`${materialId}-${topic.title || 'topic'}-${idx}`} className="rounded-md border border-primary-500/20 bg-dark-bg/60 p-3">
                      <p className="text-[11px] uppercase tracking-wide text-primary-300">Topic {topic.topicNumber}</p>
                      <p className="text-sm font-semibold text-dark-text-primary">{topic.title || `Topic ${idx + 1}`}</p>
                      {topic.summary && <p className="text-xs text-dark-text-muted mt-1">{topic.summary}</p>}
                      {Array.isArray(topic.subtopics) && topic.subtopics.length > 0 && (
                        <ul className="mt-2 list-disc pl-5 text-xs text-dark-text-muted space-y-1">
                          {topic.subtopics.slice(0, 4).map((sub, subIdx) => (
                            <li key={`${materialId}-${topic.title || 'topic'}-sub-${subIdx}`}>
                              {typeof sub === 'string' ? sub : sub?.title || sub?.narration || `Subtopic ${subIdx + 1}`}
                            </li>
                          ))}
                          {topic.subtopics.length > 4 && (
                            <li className="text-[11px] italic text-primary-200">
                              +{topic.subtopics.length - 4} more subtopic{topic.subtopics.length - 4 > 1 ? 's' : ''}
                            </li>
                          )}
                        </ul>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderTopicGenerationPanel = () => {
    const materialIds = Object.keys(generatedTopicsByMaterial);

    return (
      <div className="space-y-4">
        <div className="rounded-xl border border-dark-border bg-dark-card/60 p-6">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-lg font-semibold">Generated Topics</h3>
                <Button variant="ghost" size="small" onClick={handleExitTopicMode}>
                  Back
                </Button>
              </div>
              <p className="text-sm text-dark-text-muted">
                {materialIds.length > 0
                  ? `Topics extracted from ${materialIds.length} chapter${materialIds.length > 1 ? 's' : ''}.`
                  : 'Select a suggestion to generate topics.'}
              </p>
              <span className="text-xs text-dark-text-muted">
                {topicExtractionLoading ? 'Generating topics…' : 'Switch suggestions to regenerate topics.'}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button size="small" onClick={() => setTopicGenerationMode(false)}>
                Exit
              </Button>
              <Button
                size="small"
                variant="primary"
                onClick={handleProceedToLectureGeneration}
                disabled={!hasAnySelectedTopics}
              >
                Merge topics
              </Button>
            </div>
          </div>

          <div className="mt-4 rounded-lg border border-dark-border/60 bg-dark-card/60 p-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <p className="text-sm font-semibold text-dark-text-primary">Add Topic Manually</p>
                <p className="text-xs text-dark-text-muted">Add any extra topics you typed or edited manually.</p>
              </div>
              <Button variant="secondary" size="small" onClick={handleManualTopicAdd}>
                Add Topic
              </Button>
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-wide text-dark-text-muted">Topic Title</label>
                <input
                  type="text"
                  value={manualTopicTitle}
                  onChange={(event) => setManualTopicTitle(event.target.value)}
                  className="input-primary w-full"
                  placeholder="Enter topic title"
                />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-wide text-dark-text-muted">Topic Summary</label>
                <input
                  type="text"
                  value={manualTopicSummary}
                  onChange={(event) => setManualTopicSummary(event.target.value)}
                  className="input-primary w-full"
                  placeholder="Optional summary"
                />
              </div>
            </div>
            <div className="mt-3 space-y-2">
              <label className="text-xs font-semibold uppercase tracking-wide text-dark-text-muted">Subtopics (one per line)</label>
              <textarea
                rows={3}
                value={manualTopicSubtopics}
                onChange={(event) => setManualTopicSubtopics(event.target.value)}
                className="input-primary w-full"
                placeholder="Enter subtopics"
              />
            </div>
          </div>

          <div className="mt-4 rounded-xl border border-primary-500/40 bg-primary-500/5 p-4 space-y-4">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-primary-100">
                  <Sparkles className="h-4 w-4" />
                  <p className="text-sm font-semibold">AI Assistant (Chat)</p>
                </div>
                <p className="text-xs text-primary-200">
                  Ask multiple prompts and keep their responses stacked for quick comparison.
                </p>
                {latestAssistantMeta?.languageLabel && (
                  <p className="text-[11px] text-primary-200/90">
                    Responding in <span className="font-semibold">{latestAssistantMeta.languageLabel}</span>
                  </p>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs text-primary-100">
                <div className="rounded-md border border-primary-500/40 px-3 py-1">
                  {assistantResponses.length} prompt{assistantResponses.length !== 1 ? 's' : ''}
                </div>
                <div className="rounded-md border border-primary-500/40 px-3 py-1">
                  {assistantSelectedCount} selected
                </div>
                <Button
                  size="small"
                  variant="primary"
                  onClick={handleAssistantAddSelected}
                  disabled={!assistantHasSelections || assistantSaving}
                  loading={assistantSaving}
                >
                  Add selected topics
                </Button>
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-wide text-primary-200">Prompt</label>
                <textarea
                  rows={3}
                  value={assistantQuery}
                  onChange={(event) => setAssistantQuery(event.target.value)}
                  className="input-primary w-full"
                  placeholder="e.g. 'Suggest 5 high-yield derivations for this physics chapter'."
                />
              </div>
              <div className="space-y-3">
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="space-y-1">
                    <label className="text-xs font-semibold uppercase tracking-wide text-primary-200">Temperature</label>
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.1"
                      value={assistantTemperature}
                      onChange={(event) => setAssistantTemperature(event.target.value)}
                      className="input-primary w-full"
                    />
                    <p className="text-[11px] text-primary-200/80">Lower = focused · Higher = creative</p>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-semibold uppercase tracking-wide text-primary-200">Plan</label>
                    <select
                      className="input-primary w-full"
                      value={assistantPlanLabel}
                      onChange={(event) => setAssistantPlanLabel(event.target.value)}
                    >
                      {assistantPlanOptions.map((option) => (
                        <option key={option || 'none'} value={option}>
                          {option ? option.toUpperCase() : 'Auto-detect'}
                        </option>
                      ))}
                    </select>
                    <p className="text-[11px] text-primary-200/80">Optional: helps enforce user limits.</p>
                  </div>
                </div>
                <Button
                  size="small"
                  variant="secondary"
                  className="w-full"
                  onClick={handleAssistantAsk}
                  loading={assistantLoading}
                  disabled={assistantLoading}
                >
                  <MessageSquare className="mr-2 h-4 w-4" /> Ask Assistant
                </Button>
              </div>
            </div>

            {assistantError && (
              <p className="text-sm text-red-300">{assistantError}</p>
            )}

            <div className="space-y-4">
              {assistantResponses.length === 0 && !assistantLoading && (
                <p className="text-xs text-primary-200/80">
                  No assistant responses yet. Ask a question to start the chat.
                </p>
              )}

              {assistantResponses.map((response, responseIndex) => {
                const suggestions = response.suggestions || [];
                const selectedIndices = assistantSelections[response.id] || [];
                const hasSuggestions = suggestions.length > 0;
                const SelectionIcon = ({ isSelected }) => (isSelected ? CheckSquare : Square);

                return (
                  <div
                    key={`assistant-response-${response.id}`}
                    className="rounded-lg border border-dark-border/50 bg-dark-card/70 p-4 space-y-3"
                  >
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                      <div className="space-y-1">
                        <p className="text-[11px] uppercase tracking-wide text-primary-200">
                          Prompt #{responseIndex + 1}
                        </p>
                        <p className="text-sm font-semibold text-dark-text-primary">
                          {response.prompt || 'Prompt'}
                        </p>
                        {response.meta?.reply && (
                          <p className="text-xs text-dark-text-muted whitespace-pre-line">
                            {response.meta.reply}
                          </p>
                        )}
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <Button
                          size="extraSmall"
                          variant="ghost"
                          onClick={() => handleAssistantSelectAll(response.id)}
                          disabled={!hasSuggestions}
                        >
                          {selectedIndices.length === suggestions.length && suggestions.length > 0
                            ? 'Clear selection'
                            : 'Select all'}
                        </Button>
                        <span className="text-xs text-primary-200">
                          {selectedIndices.length}/{suggestions.length} selected
                        </span>
                      </div>
                    </div>

                    {hasSuggestions ? (
                      <div className="space-y-3">
                        {suggestions.map((suggestion, index) => {
                          const isSelected = selectedIndices.includes(index);
                          const IconComponent = SelectionIcon({ isSelected });
                          return (
                            <div
                              key={`${response.id}-${suggestion.title || 'suggestion'}-${index}`}
                              className={`rounded-lg border px-4 py-3 transition ${isSelected ? 'border-primary-500 bg-primary-500/10' : 'border-dark-border/50 bg-dark-card/60'}`}
                            >
                              <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                                <div className="space-y-1">
                                  <p className="text-[11px] uppercase tracking-wide text-primary-200">
                                    Suggestion {index + 1}
                                  </p>
                                  <p className="text-sm font-semibold text-dark-text-primary">
                                    {suggestion.title || `Suggestion ${index + 1}`}
                                  </p>
                                  {suggestion.summary && (
                                    <p className="text-xs text-dark-text-muted">{suggestion.summary}</p>
                                  )}
                                  {suggestion.supporting_quote && (
                                    <blockquote className="border-l-2 border-primary-400/60 pl-3 text-[11px] italic text-primary-200">
                                      {suggestion.supporting_quote}
                                    </blockquote>
                                  )}
                                </div>
                                <button
                                  type="button"
                                  onClick={() => handleAssistantToggleSelection(response.id, index)}
                                  className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-medium transition ${isSelected
                                    ? 'border-primary-500/60 bg-primary-500/10 text-primary-100'
                                    : 'border-dark-border/60 bg-dark-card/40 text-dark-text-muted hover:text-dark-text-primary'
                                  }`}
                                >
                                  <IconComponent className="h-3.5 w-3.5" />
                                  {isSelected ? 'Selected' : 'Select'}
                                </button>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="text-xs text-dark-text-muted">No suggestions returned for this prompt.</p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-4 space-y-4">
            {topicExtractionLoading ? (
              <div className="flex items-center gap-2 text-sm text-dark-text-muted">
                <Loader2 className="h-4 w-4 animate-spin" />
                Extracting topics...
              </div>
            ) : generatedTopicsError ? (
              <p className="text-sm text-red-400">{generatedTopicsError}</p>
            ) : materialIds.length > 0 ? (
              materialIds.map((materialId) => renderTopicsForMaterial(Number(materialId)))
            ) : (
              <p className="text-sm text-dark-text-muted">Generate topics to see them listed here.</p>
            )}
          </div>
        </div>

        {/* Previously Extracted Topics section removed per request */}
      </div>
    );
  };

  const renderMergedTopicsScreen = () => {
    const content = mergedTopicsPreview();

    if (!content) {
      return (
        <div className="rounded-xl border border-dark-border bg-dark-card/60 p-6 space-y-4">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className="text-lg font-semibold">Merged Topics</h3>
              <p className="text-sm text-dark-text-muted">Select topics from chapters to create a merged view.</p>
            </div>
            <Button size="small" onClick={() => setShowMergedPreview(false)}>
              Back to selection
            </Button>
          </div>
          <p className="text-sm text-dark-text-muted">No topics selected yet.</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="text-lg font-semibold text-dark-text-primary">Merged Topics</h3>
            <p className="text-sm text-dark-text-muted">Review merged topics before generating the lecture.</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button size="small" variant="ghost" onClick={() => setShowMergedPreview(false)}>
              Back to topic selection
            </Button>
            <Button size="small" variant="primary" onClick={handleGenerateMergedLecture}>
              Generate lecture
            </Button>
          </div>
        </div>
        {content}
      </div>
    );
  };

  const renderUploadPanel = () => (
    <div className="rounded-xl border border-dark-border bg-dark-card/60 p-6">
      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
        <div>
          <h3 className="text-lg font-semibold">Upload Chapter PDF</h3>
          <p className="text-sm text-dark-text-muted">Upload a PDF for the selected class, subject, and chapter.</p>
        </div>
        {chapterTitle && (
          <div className="text-right text-xs text-dark-text-muted">
            <p>
              Chapter: <span className="text-dark-text-primary">{chapterTitle}</span>
            </p>
            <p>
              Class {uploadForm.std || '—'} · {uploadForm.subject || '—'} · {uploadForm.sem || 'Semester'} · {uploadForm.board || 'Board'}
            </p>
          </div>
        )}
      </div>

      <div className="mt-4">
        <FileUploader
          key={uploaderResetKey}
          label="Chapter PDF"
          description="Click to upload or drag a PDF (max 15MB)."
          accept="application/pdf"
          multiple={false}
          maxFiles={1}
          maxSize={PDF_MAX_SIZE_BYTES}
          onFilesChange={setPdfFiles}
        />
      </div>

      <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-xs text-dark-text-muted">
          Supports only PDF files. Ensure class, subject, semester, board, and chapter are filled before uploading.
        </p>
        <Button variant="primary" size="small" onClick={handleUpload} loading={uploading} disabled={!pdfFiles.length || uploading}>
          Upload Chapter PDF
        </Button>
      </div>
    </div>
  );

  const renderStandardAddPanel = () => (
    <>
      <div className="rounded-xl border border-dark-border bg-dark-card/60 p-6">
        <div>
          <h3 className="text-lg font-semibold">Add Class</h3>
          <p className="text-sm text-dark-text-muted">Class</p>
        </div>
        <div className="mt-4">
          <InputField
            label="Class"
            placeholder="Enter Class"
            value={uploadForm.std}
            onChange={handleUploadFieldChange('std')}
            containerClassName="w-full"
          />
        </div>
      </div>

      <div className="rounded-xl border border-dark-border bg-dark-card/60 p-6">
        <div>
          <h3 className="text-lg font-semibold">Add Subject</h3>
          <p className="text-sm text-dark-text-muted">Subject details for this chapter.</p>
        </div>
        <div className="mt-4">
          <InputField
            label="Subject Name"
            placeholder="Enter Subject Name"
            value={uploadForm.subject}
            onChange={handleUploadFieldChange('subject')}
            containerClassName="w-full"
          />
        </div>

        {!hasCompleteUploadContext && (
          <p className="mt-4 text-xs text-dark-text-muted">Provide class and subject details to enable uploads.</p>
        )}
      </div>

      <div className="rounded-xl border border-dark-border bg-dark-card/60 p-6">
        <div>
          <h3 className="text-lg font-semibold">Chapter Name</h3>
          <p className="text-sm text-dark-text-muted">Add the chapter title you want to manage.</p>
        </div>
        <div className="mt-4">
          <InputField
            label="Chapter"
            placeholder="Enter Chapter Name"
            value={chapterTitle}
            onChange={(event) => setChapterTitle(event.target.value)}
            containerClassName="w-full"
          />
        </div>
        <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs text-dark-text-muted">
            Save the entered class, subject, and chapter to fetch matching suggestions from the library.
          </p>
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="small"
              onClick={handleSaveUploadContext}
              disabled={!hasSuggestionContext || suggestionsUnlocked}
            >
              {suggestionsUnlocked ? 'Saved' : 'Save context'}
            </Button>
            <Button
              variant="ghost"
              size="small"
              onClick={handleSkipContext}
              disabled={!hasSuggestionContext}
            >
              Skip
            </Button>
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-dark-border bg-dark-card/60 p-6">
        <div>
          <h3 className="text-lg font-semibold">Chapter Suggestions</h3>
          <p className="text-sm text-dark-text-muted">
            Recommended chapters based on your existing library for the selected class and subject.
          </p>
        </div>
        <div className="mt-4 rounded-xl border border-dark-border/70 bg-dark-bg/30 p-4">
          {!hasSuggestionContext ? (
            <p className="text-sm text-dark-text-muted">Enter class, subject, and chapter name to enable suggestions.</p>
          ) : !suggestionsUnlocked ? (
            <p className="text-sm text-dark-text-muted">Click "Save context" above to load chapter suggestions for the entered details.</p>
          ) : suggestionsLoading ? (
            <div className="flex items-center gap-2 text-sm text-dark-text-muted">
              <Loader2 className="h-4 w-4 animate-spin" />
              Fetching chapter suggestions...
            </div>
          ) : suggestedMaterials.length === 0 ? (
            <p className="text-sm text-dark-text-muted">No existing chapters found for this class and subject. Add your first one below.</p>
          ) : (
            <div className="space-y-3">
              {suggestedMaterials.slice(0, 5).map((material) => (
                <div key={material.id} className="rounded-lg border border-dark-border/70 bg-dark-card/40 px-4 py-3">
                  <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <p className="text-sm font-semibold text-dark-text-primary">{material.chapter_number || 'Untitled chapter'}</p>
                      <p className="text-xs text-dark-text-muted">Class {material.std || '—'} · {material.subject || '—'}</p>
                      <p className="text-xs text-dark-text-muted">Semester {material.sem || '—'} · {material.board || '—'}</p>
                    </div>
                    <div className="flex flex-col items-start gap-2 sm:items-end">
                      <p className="text-xs text-dark-text-muted">{formatDate(material.created_at)}</p>
                      {material.file_path ? (
                        <a
                          href={getFileUrl(material.file_path)}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 rounded-md border border-dark-border px-2 py-1 text-xs text-dark-text-primary hover:border-primary-500 hover:text-primary-100"
                        >
                          <Eye className="w-3.5 h-3.5" />
                          View PDF
                        </a>
                      ) : (
                        <span className="text-xs text-dark-text-muted">PDF unavailable</span>
                      )}
                      <button
                        type="button"
                        onClick={() => handleToggleSuggestionSelection(material.id)}
                        className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs font-medium transition ${selectedSuggestionIds.includes(material.id)
                            ? 'border-primary-500 bg-primary-500/10 text-primary-100'
                            : 'border-dark-border/70 bg-dark-card/60 text-dark-text-muted hover:text-dark-text-primary'
                          }`}
                      >
                        {selectedSuggestionIds.includes(material.id) ? (
                          <CheckSquare className="w-3.5 h-3.5" />
                        ) : (
                          <Square className="w-3.5 h-3.5" />
                        )}
                        <span>{selectedSuggestionIds.includes(material.id) ? 'Selected' : 'Select'}</span>
                      </button>
                    </div>
                  </div>
                </div>
              ))}
              {suggestedMaterials.length > 5 && (
                <p className="text-xs text-dark-text-muted">Showing first 5 matches out of {suggestedMaterials.length}.</p>
              )}
            </div>
          )}
        </div>
        <div className="mt-4 flex justify-end">
          <Button
            variant="primary"
            size="small"
            onClick={handleProceedWithSelectedChapters}
            disabled={!selectedSuggestionIds.length}
          >
            Next
          </Button>
        </div>
      </div>
    </>
  );

  const renderRecentUploadsPanel = () => (
    <div className="rounded-xl border border-dark-border bg-dark-card/60 p-6">
      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
        <div>
          <h3 className="text-lg font-semibold">Recently Uploaded</h3>
          <p className="text-sm text-dark-text-muted">Your latest uploads appear here for quick confirmation.</p>
        </div>
      </div>

      <div className="mt-6 overflow-hidden rounded-lg border border-dark-border/60">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-dark-border">
            <thead className="bg-dark-card/80 text-xs uppercase tracking-wide text-dark-text-muted">
              <tr>
                <th className="px-6 py-3 text-left">Chapter</th>
                <th className="px-6 py-3 text-left">Subject</th>
                <th className="px-6 py-3 text-left">Class</th>
                <th className="px-6 py-3 text-left">Semester</th>
                <th className="px-6 py-3 text-left">Board</th>
                <th className="px-6 py-3 text-left">Extracted Topics</th>
                <th className="px-6 py-3 text-left">Uploaded</th>
                <th className="px-6 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-border/60 text-sm">
              {recentUploads.length > 0 ? (
                recentUploads.map((material) => (
                  <tr key={material.id} className="transition hover:bg-dark-card/40">
                    <td className="px-6 py-4">
                      <div className="space-y-1">
                        <p className="font-medium text-dark-text-primary">Chapter {material.chapter_number || '—'}</p>
                        <p className="text-xs text-dark-text-muted">{material.file_name || 'PDF resource'}</p>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-dark-text-primary">{material.subject || '—'}</td>
                    <td className="px-6 py-4">{material.std || '—'}</td>
                    <td className="px-6 py-4">{material.sem || '—'}</td>
                    <td className="px-6 py-4">
                      <span className="rounded-full bg-dark-card px-3 py-1 text-xs text-dark-text-muted">{material.board || '—'}</span>
                    </td>
                    <td className="px-6 py-4">
                      {recentTopicsMap[material.id]?.length ? (
                        <div className="space-y-1">
                          {recentTopicsMap[material.id].slice(0, 3).map((topic, index) => (
                            <p key={`${material.id}-topic-${index}`} className="text-xs text-dark-text-primary">
                              • {topic.title || `Topic ${index + 1}`}
                            </p>
                          ))}
                          {recentTopicsMap[material.id].length > 3 && (
                            <p className="text-[11px] text-dark-text-muted">
                              +{recentTopicsMap[material.id].length - 3} more
                            </p>
                          )}
                        </div>
                      ) : (
                        <span className="text-xs text-dark-text-muted">No topics yet</span>
                      )}
                    </td>
                    <td className="px-6 py-4 space-y-1">
                      <p className="text-sm text-dark-text-primary">{formatDate(material.created_at)}</p>
                      <p className="text-xs text-dark-text-muted">{formatFileSize(material.file_size)}</p>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center justify-end gap-2">
                        <a
                          href={getFileUrl(material.file_path)}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-2 rounded-lg border border-dark-border px-3 py-2 text-xs text-dark-text-primary hover:border-primary-500 hover:text-primary-100"
                        >
                          <Eye className="w-4 h-4" />
                          View
                        </a>
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={8} className="px-6 py-10 text-center text-dark-text-muted">
                    <div className="space-y-3">
                      <p className="font-medium text-dark-text-primary">No recent uploads yet</p>
                      <p className="text-xs text-dark-text-muted">Upload a PDF to see it appear here instantly.</p>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );


  if (loadingDashboard) {
    return (
      <div className="min-h-screen bg-dark-bg flex items-center justify-center">
        <div className="spinner w-8 h-8" />
      </div>
    );
  }

  const chapterMetrics = dashboardData?.chapter_metrics || {};
  const startIndex = filteredCount === 0 ? 0 : (currentPage - 1) * PAGE_SIZE + 1;
  const endIndex = filteredCount === 0 ? 0 : Math.min(startIndex + PAGE_SIZE - 1, filteredCount);

  return (
    <div className="min-h-screen bg-dark-bg text-dark-text-primary">
      <header className="bg-dark-card border-b border-dark-border px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-primary-600 rounded-full flex items-center justify-center">
              <BookPlus className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-dark-text-primary">Chapter Dashboard</h1>
              <p className="text-sm text-dark-text-muted">Plan, upload, and organise chapter PDFs effortlessly.</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="small"
              icon={Filter}
              onClick={handleToggleFilterPanel}
              className="hidden sm:inline-flex"
            >
              {isFilterPanelOpen ? 'Hide Filters' : 'Show Filters'}
            </Button>
            <button className="p-2 text-dark-text-muted hover:text-dark-text-primary transition">
              <Bell className="w-5 h-5" />
            </button>
            <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
              <span className="text-white text-sm font-medium">
                {user?.name?.charAt(0)?.toUpperCase() || 'C'}
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        <aside className="hidden lg:flex w-72 flex-col border-r border-dark-border bg-dark-card/60 p-6">
          <nav className="space-y-1">
            {NAV_ITEMS.map(({ id, label, description, icon: Icon }) => {
              const isActive = activeSection === id;
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => handleNavClick(id)}
                  className={`w-full rounded-xl border px-4 py-3 text-left transition ${isActive
                      ? 'border-primary-500 bg-primary-600/20 text-primary-100'
                      : 'border-transparent text-dark-text-muted hover:border-dark-border hover:text-dark-text-primary'
                    }`}
                >
                  <div className="flex items-center gap-3">
                    <Icon className={`w-5 h-5 ${isActive ? 'text-primary-300' : 'text-dark-text-muted'}`} />
                    <div>
                      <p className="text-sm font-medium">{label}</p>
                      <p className="text-xs text-dark-text-muted/80">{description}</p>
                    </div>
                  </div>
                </button>
              );
            })}
          </nav>

          <div className="mt-auto pt-6">
            <div className="rounded-lg border border-dark-border bg-dark-card/70 p-4 space-y-3">
              <p className="text-xs uppercase tracking-wide text-dark-text-muted">Finished for the day?</p>
              <Button variant="danger" icon={LogOut} fullWidth onClick={handleLogout}>
                Logout
              </Button>
            </div>
          </div>
        </aside>

        <main className="flex-1 p-6 space-y-6">
          <div className="lg:hidden card p-4">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
                const isActive = activeSection === id;
                return (
                  <button
                    key={id}
                    type="button"
                    onClick={() => handleNavClick(id)}
                    className={`flex items-center justify-center gap-2 rounded-xl border px-4 py-2 text-sm transition ${isActive
                        ? 'border-primary-500 bg-primary-600/20 text-primary-100'
                        : 'border-dark-border/60 bg-dark-card text-dark-text-muted hover:text-dark-text-primary'
                      }`}
                  >
                    <Icon className="w-4 h-4" />
                    {label}
                  </button>
                );
              })}
            </div>
          </div>

          {activeSection === 'overview' ? (
            <div className="space-y-6">
              {renderChapterOverview()}
            </div>
          ) : activeSection === 'add' ? (
            <motion.div className="card p-0 overflow-hidden" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
              <div className="flex items-center justify-between border-b border-dark-border px-6 py-5">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 bg-primary-600 rounded-full flex items-center justify-center">
                    <BookPlus className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold">Add Chapter Management</h2>
                    <p className="text-sm text-dark-text-muted">Add class, subject and chapter details, then upload a PDF.</p>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button variant="ghost" size="small" onClick={resetUploadForm}>
                    Cancel
                  </Button>
                  <Button
                    variant="primary"
                    size="small"
                    onClick={handleNextClick}
                    loading={uploading}
                    disabled={
                      topicGenerationMode
                        ? !hasSelectedTopics || topicExtractionLoading
                        : !hasSuggestionContext
                    }
                  >
                    Next
                  </Button>
                </div>
              </div>

              <div className="space-y-6 px-6 py-6">
                {skipContextMode ? (
                  renderSkipUploadPanel()
                ) : (
                  <>
                    {topicGenerationMode
                      ? showMergedPreview
                        ? renderMergedTopicsScreen()
                        : renderTopicGenerationPanel()
                      : renderStandardAddPanel()}
                    {renderRecentUploadsPanel()}
                  </>
                )}
              </div>
            </motion.div>
          ) : (
            <motion.div className="rounded-xl border border-dark-border bg-dark-card/60 p-6 space-y-4" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h3 className="text-lg font-semibold">Chapter Material Library</h3>
                  <p className="text-sm text-dark-text-muted">
                    {filteredCount > 0
                      ? `Showing ${filteredCount} curated PDF${filteredCount > 1 ? 's' : ''} for the selected filters.`
                      : 'Start by choosing a class context to discover relevant PDFs.'}
                  </p>
                </div>
                <div className="flex flex-col sm:flex-row sm:items-center gap-3">
                  <div className="relative w-full sm:w-64">
                    <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-dark-text-muted" />
                    <input
                      value={searchQuery}
                      onChange={handleSearchChange}
                      placeholder="Search by subject, class or chapter"
                      className="input-primary w-full pl-10"
                    />
                  </div>
                  <Button
                    variant="secondary"
                    size="small"
                    icon={RefreshCw}
                    onClick={handleRefresh}
                    loading={materialsLoading}
                  >
                    Refresh
                  </Button>
                </div>
              </div>
            </motion.div>
          )}
        </main>
      </div>
    </div>
  );
};

export default ChapterDashboard;
