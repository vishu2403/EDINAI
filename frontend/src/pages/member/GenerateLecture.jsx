/**
 * Generate Lecture - Review selected topics and start lecture creation flow
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { useLocation, useNavigate } from 'react-router-dom';
import Button from '../../components/ui/Button';
import {
  ArrowLeft,
  BookOpen,
  Sparkles,
  Loader2,
  Clipboard,
  FileDown,
  ChevronLeft,
  ChevronRight,
  Save,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { lectureAPI } from '../../utils/api';

const LECTURE_STRUCTURE = [
  { key: 'introduction', name: 'Introduction', slideNumbers: [1] },
  { key: 'key_concepts', name: 'Key Concepts', slideNumbers: [2, 3] },
  { key: 'deep_teaching', name: 'Deep Teaching with Examples', slideNumbers: [4, 5, 6, 7] },
  { key: 'practical_applications', name: 'Practical Applications', slideNumbers: [8] },
  { key: 'quiz_reflection', name: 'Quiz & Reflection', slideNumbers: [9] },
];

const slideMetadata = {
  1: {
    heading: 'Slide 1 • Introduction',
    description: 'Warm welcome and high-level overview of today\'s lecture.',
    accent: 'from-primary-500/40 via-primary-500/10 to-transparent',
  },
  2: {
    heading: 'Slide 2 • Key Concepts (Part 1)',
    description: 'First set of core ideas presented as concise bullet points.',
    accent: 'from-emerald-500/30 via-emerald-500/10 to-transparent',
  },
  3: {
    heading: 'Slide 3 • Key Concepts (Part 2)',
    description: 'Second set of essential concepts to set up deeper teaching.',
    accent: 'from-emerald-500/30 via-emerald-500/10 to-transparent',
  },
  4: {
    heading: 'Slide 4 • Deep Teaching Block',
    description: 'Theory plus real-world example woven together (600-1000 words).',
    accent: 'from-violet-500/30 via-violet-500/10 to-transparent',
  },
  5: {
    heading: 'Slide 5 • Deep Teaching Block',
    description: 'Continuation of storytelling-style teaching with case studies.',
    accent: 'from-violet-500/30 via-violet-500/10 to-transparent',
  },
  6: {
    heading: 'Slide 6 • Deep Teaching Block',
    description: 'Explores an additional concept with immediate practical context.',
    accent: 'from-violet-500/30 via-violet-500/10 to-transparent',
  },
  7: {
    heading: 'Slide 7 • Deep Teaching Block',
    description: 'Final theory + example pairing to round out the lesson.',
    accent: 'from-violet-500/30 via-violet-500/10 to-transparent',
  },
  8: {
    heading: 'Slide 8 • Practical Applications',
    description: 'Actionable guidance on how students can use the knowledge.',
    accent: 'from-amber-500/30 via-amber-500/10 to-transparent',
  },
  9: {
    heading: 'Slide 9 • Quiz & Reflection',
    description: 'Five thoughtful questions to reinforce the material.',
    accent: 'from-sky-500/30 via-sky-500/10 to-transparent',
  },
};

const slideThemeConfig = {
  1: {
    borderClass: 'border-sky-500 bg-black/85',
    chipClass: 'text-sky-200 bg-sky-500/20',
    headerBadge: 'bg-sky-500 text-white',
    description: 'Introduction',
  },
  2: {
    borderClass: 'border-amber-500 bg-black/85',
    chipClass: 'text-amber-200 bg-amber-500/20',
    headerBadge: 'bg-amber-500 text-amber-50',
    description: 'Key Concepts - Part 1',
  },
  3: {
    borderClass: 'border-amber-500 bg-black/85',
    chipClass: 'text-amber-200 bg-amber-500/20',
    headerBadge: 'bg-amber-500 text-amber-50',
    description: 'Key Concepts - Part 2',
  },
  4: {
    borderClass: 'border-violet-500 bg-black/85',
    chipClass: 'text-violet-200 bg-violet-500/20',
    headerBadge: 'bg-violet-500 text-violet-50',
    description: 'Deep Teaching with Examples - Part 1',
  },
  5: {
    borderClass: 'border-violet-500 bg-black/85',
    chipClass: 'text-violet-200 bg-violet-500/20',
    headerBadge: 'bg-violet-500 text-violet-50',
    description: 'Deep Teaching with Examples - Part 2',
  },
  6: {
    borderClass: 'border-violet-500 bg-black/85',
    chipClass: 'text-violet-200 bg-violet-500/20',
    headerBadge: 'bg-violet-500 text-violet-50',
    description: 'Deep Teaching with Examples - Part 3',
  },
  7: {
    borderClass: 'border-violet-500 bg-black/85',
    chipClass: 'text-violet-200 bg-violet-500/20',
    headerBadge: 'bg-violet-500 text-violet-50',
    description: 'Deep Teaching with Examples - Part 4',
  },
  8: {
    borderClass: 'border-emerald-500 bg-black/85',
    chipClass: 'text-emerald-200 bg-emerald-500/20',
    headerBadge: 'bg-emerald-500 text-emerald-50',
    description: 'Practical Applications',
  },
  9: {
    borderClass: 'border-rose-500 bg-black/85',
    chipClass: 'text-rose-200 bg-rose-500/20',
    headerBadge: 'bg-rose-500 text-rose-50',
    description: 'Quiz & Reflection',
  },
};

const getSlideMeta = (number) => {
  if (slideMetadata[number]) {
    return slideMetadata[number];
  }

  return {
    heading: `Slide ${number}`,
    description: 'AI-generated slide content.',
    accent: 'from-dark-border/40 via-dark-border/10 to-transparent',
  };
};

const renderNarration = (
  narration,
  paragraphClassName = 'text-sm leading-relaxed text-white/85',
) => {
  if (!narration) {
    return null;
  }

  return narration
    .split(/\n{2,}|\r?\n/)
    .filter(Boolean)
    .map((paragraph, idx) => (
      <p key={idx} className={paragraphClassName}>
        {paragraph.trim()}
      </p>
    ));
};

const DEFAULT_MIN_DURATION = 5;
const DEFAULT_MAX_DURATION = 180;

const GenerateLecture = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [cachedSelection, setCachedSelection] = useState(() => {
    try {
      const cached = localStorage.getItem('inai_pending_lecture_selection');
      return cached ? JSON.parse(cached) : null;
    } catch (error) {
      console.error('Failed to parse pending lecture selection:', error);
      return null;
    }
  });

  const [selection, setSelection] = useState(() => location.state || cachedSelection || null);

  useEffect(() => {
    if (location.state) {
      setSelection(location.state);
      localStorage.removeItem('inai_pending_lecture_selection');
      setCachedSelection(null);
    }
  }, [location.state]);

  useEffect(() => {
    if (!selection && cachedSelection) {
      setSelection(cachedSelection);
      localStorage.removeItem('inai_pending_lecture_selection');
      setCachedSelection(null);
    }
  }, [selection, cachedSelection]);

  const { material, selectedTopicIndices = [], topics = [] } = selection || {};
  const [isGenerating, setIsGenerating] = useState(false);
  const [lecture, setLecture] = useState(null);
  const [activeSlideIndex, setActiveSlideIndex] = useState(0);
  const [editedNarration, setEditedNarration] = useState('');
  const [editedBullets, setEditedBullets] = useState('');
  const [initialLecture, setInitialLecture] = useState(null);
  const editPanelRef = useRef(null);
  const [configLoading, setConfigLoading] = useState(true);
  const [configError, setConfigError] = useState(null);
  const [lectureConfig, setLectureConfig] = useState({
    languages: [],
    durations: [],
    default_language: '',
    default_duration: null,
  });
  const [language, setLanguage] = useState('');
  const [duration, setDuration] = useState(null);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setConfigLoading(true);
        const response = await lectureAPI.getConfig();
        if (!response?.status) {
          throw new Error(response?.message || 'Config load failed');
        }

        const data = response.data || {};
        setLectureConfig({
          languages: data.languages || [],
          durations: data.durations || [],
          default_language: (data.default_language || '').trim(),
          default_duration: data.default_duration || null,
        });

        if (!language && (data.selected_language || data.default_language)) {
          setLanguage((data.selected_language || data.default_language || '').trim());
        }

        if (
          !duration &&
          (data.selected_duration || data.default_duration || data.durations?.length)
        ) {
          setDuration(data.selected_duration || data.default_duration || data.durations?.[0] || null);
        }
      } catch (error) {
        console.error('Lecture config fetch error:', error);
        setConfigError(error.response?.data?.detail || error.message || 'Failed to load lecture settings');
        toast.error('Lecture settings load failed');
      } finally {
        setConfigLoading(false);
      }
    };

    fetchConfig();
  }, []);

  useEffect(() => {
    if (selection) {
      return undefined;
    }

    const timeoutId = setTimeout(() => {
      toast.error('Please select topics before generating a lecture.');
      navigate('/chapter-dashboard', { replace: true });
    }, 2000);

    return () => clearTimeout(timeoutId);
  }, [selection, navigate]);

  const topicIndices = useMemo(() => {
    if (!selectedTopicIndices) return [];
    return Array.isArray(selectedTopicIndices)
      ? selectedTopicIndices
      : Array.from(selectedTopicIndices);
  }, [selectedTopicIndices]);

  const selectedTopics = useMemo(() => {
    if (!topics || topicIndices.length === 0) return [];

    const normalizeId = (value) => {
      if (value === undefined || value === null) {
        return null;
      }
      const stringValue = String(value).trim();
      return stringValue || null;
    };

    return topicIndices
      .map((topicIndex) => {
        const topic = topics[topicIndex];
        if (!topic) return null;

        if (typeof topic === 'string') {
          return {
            index: topicIndex,
            title: topic,
            summary: '',
            subtopics: [],
            topicId: null,
            suggestionTopicId: null,
          };
        }

        const topicId =
          topic.topic_id ??
          topic.topicId ??
          topic.id ??
          topic.topicID ??
          topic.topic_uuid ??
          topic.uuid ??
          null;
        const suggestionTopicId =
          topic.suggestion_topic_id ??
          topic.suggestionTopicId ??
          topic.suggestion_id ??
          topic.suggestionId ??
          topic.suggestionTopicID ??
          null;

        return {
          index: topicIndex,
          title: topic.title || `Topic ${topicIndex + 1}`,
          summary: topic.summary || '',
          subtopics: Array.isArray(topic.subtopics) ? topic.subtopics : [],
          topicId: normalizeId(topicId),
          suggestionTopicId: normalizeId(suggestionTopicId),
        };
      })
      .filter(Boolean);
  }, [topics, topicIndices]);

  const selectedTopicIds = useMemo(
    () =>
      selectedTopics
        .map((topic) => (
          topic && typeof topic === 'object'
            ? topic.topicId || topic.suggestionTopicId || null
            : null
        ))
        .filter(Boolean),
    [selectedTopics],
  );

  if (!selection || !material) {
    return (
      <div className="min-h-screen bg-dark-bg text-dark-text-primary flex flex-col items-center justify-center gap-4">
        <Loader2 className="h-10 w-10 animate-spin text-primary-400" />
        <p className="text-sm text-dark-text-muted">
          Loading your selected topics... Please wait or return to the dashboard to pick topics again.
        </p>
        <Button variant="secondary" size="small" onClick={() => navigate('/chapter-dashboard', { replace: true })}>
          Back to Chapter Dashboard
        </Button>
      </div>
    );
  }

  const handleBeginGeneration = async () => {
    if (isGenerating) return;

    if (!language) {
      toast.error('Please select a lecture language');
      return;
    }

    if (!duration) {
      toast.error('Please enter duration in minutes');
      return;
    }

    if (!topicIndices.length) {
      toast.error('Please select topics before generating a lecture.');
      return;
    }

    try {
      setIsGenerating(true);
      toast.loading('Generating lecture content...', { id: 'lecture-generation' });

      const response = await lectureAPI.generateLecture({
        materialId: material.id,
        selectedTopicIndices: topicIndices,
        selectedTopicIds,
        language: language || material.language || material.detected_language,
        duration,
        title: `${material.subject} Lecture`,
      });

      if (!response?.status) {
        throw new Error(response?.message || 'Lecture generation failed.');
      }

      const lectureData = response.data?.lecture;
      setLecture(lectureData);
      setInitialLecture(lectureData ? JSON.parse(JSON.stringify(lectureData)) : null);
      setActiveSlideIndex(0);
      toast.success('Lecture generated successfully!', { id: 'lecture-generation' });
    } catch (error) {
      console.error('Lecture generation error:', error);
      toast.error(error.response?.data?.detail || error.message || 'Failed to generate lecture', {
        id: 'lecture-generation',
      });
    } finally {
      setIsGenerating(false);
    }
  };

  useEffect(() => {
    if (!material || lecture || topicIndices.length === 0) {
      return;
    }

    if (configLoading || configError || !language || !duration) {
      return;
    }

    // Previously auto-triggered generation; now we wait for user to click the button
  }, [material, lecture, topicIndices, configLoading, configError, language, duration]);

  const handleDownloadLectureJson = () => {
    if (!lecture) return;

    const blob = new Blob([JSON.stringify(lecture, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${lecture.title?.replace(/\s+/g, '_') || 'lecture'}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleCopySummary = async () => {
    if (!lecture?.summary) {
      toast.error('No summary available to copy.');
      return;
    }

    try {
      await navigator.clipboard.writeText(lecture.summary);
      toast.success('Lecture summary copied to clipboard');
    } catch (error) {
      console.error('Copy summary failed:', error);
      toast.error('Unable to copy summary.');
    }
  };

  const handleSaveLectureDetails = () => {
    if (!lecture) {
      toast.error('Generate a lecture before saving.');
      return;
    }

    try {
      localStorage.setItem('inai_latest_lecture', JSON.stringify(lecture));
      toast.success('Lecture details saved locally.');
    } catch (error) {
      console.error('Save lecture failed:', error);
      toast.error('Unable to save lecture details.');
    }
  };

  const handleApplySlideEdits = () => {
    if (!lecture) return;

    const slides = Array.isArray(lecture.slides) ? [...lecture.slides] : [];
    if (!slides[activeSlideIndex]) return;

    const cleanedBullets = editedBullets
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    slides[activeSlideIndex] = {
      ...slides[activeSlideIndex],
      narration: editedNarration.trim(),
      bullets: cleanedBullets,
    };

    setLecture({
      ...lecture,
      slides,
    });
    toast.success('Slide updated');
  };

  const handleRestoreSlide = () => {
    const originalSlides = initialLecture?.slides || [];
    if (!originalSlides.length || !originalSlides[activeSlideIndex]) {
      toast.error('Original slide not available.');
      return;
    }

    const originalSlide = originalSlides[activeSlideIndex];
    const originalBullets = Array.isArray(originalSlide.bullets)
      ? originalSlide.bullets.filter(Boolean)
      : [];

    setEditedNarration(originalSlide.narration || '');
    setEditedBullets(originalBullets.join('\n'));

    setLecture((prevLecture) => {
      if (!prevLecture) return prevLecture;
      const prevSlides = Array.isArray(prevLecture.slides) ? [...prevLecture.slides] : [];
      prevSlides[activeSlideIndex] = JSON.parse(JSON.stringify(originalSlide));

      return {
        ...prevLecture,
        slides: prevSlides,
      };
    });

    toast.success('Slide restored to original content');
  };

  const handleScrollToEdit = () => {
    if (editPanelRef.current) {
      editPanelRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const handleFinalizeLecture = () => {
    handleSaveLectureDetails();
  };

  useEffect(() => {
    if (!lecture) {
      setActiveSlideIndex(0);
    }
  }, [lecture]);

  useEffect(() => {
    if (!lecture) return;
    const slides = lecture.slides || [];
    const activeSlide = slides[activeSlideIndex];
    if (!activeSlide) return;

    setEditedNarration(activeSlide.narration || '');
    const bullets = Array.isArray(activeSlide.bullets)
      ? activeSlide.bullets.filter(Boolean)
      : [];
    setEditedBullets(bullets.join('\n'));
  }, [lecture, activeSlideIndex]);

  const goToSlide = (index) => {
    if (!lecture) return;
    const slides = lecture.slides || [];
    if (index >= 0 && index < slides.length) {
      setActiveSlideIndex(index);
    }
  };

  return (
    <div className="min-h-screen bg-dark-bg text-dark-text-primary">
      <header className="bg-dark-card border-b border-dark-border px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="small"
              icon={ArrowLeft}
              onClick={() => navigate(-1)}
              className="text-dark-text-muted hover:text-dark-text-primary"
            >
              Back
            </Button>
            <div className="w-10 h-10 bg-primary-600 rounded-full flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold">Generate Lecture</h1>
              <p className="text-sm text-dark-text-muted">
                Review your selected topics before generating the lecture content.
              </p>
            </div>
          </div>
          <Button
            variant="secondary"
            size="small"
            onClick={() => navigate('/chapter-dashboard')}
          >
            Go to Dashboard
          </Button>
        </div>
      </header>

      <main className="p-6 space-y-6">
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-2xl border border-dark-border/60 bg-dark-card/70 p-5 shadow-lg">
            <div className="flex items-center gap-3 mb-4">
              <div className="h-10 w-10 rounded-full bg-[#1f2937] flex items-center justify-center text-rose-200">⏱️</div>
              <div>
                <p className="text-sm font-semibold text-white">Set Time For Single Chapter</p>
                <p className="text-xs text-dark-text-muted">Select the duration for this lecture</p>
              </div>
            </div>

            <label className="text-xs font-semibold uppercase tracking-wide text-dark-text-muted">Set Minute</label>
            <input
              type="number"
              min={DEFAULT_MIN_DURATION}
              max={DEFAULT_MAX_DURATION}
              value={duration ?? ''}
              onChange={(event) => setDuration(Number(event.target.value) || '')}
              className="mt-2 w-full rounded-xl border border-dark-border/60 bg-dark-card/70 px-4 py-3 text-sm text-dark-text-primary focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
              placeholder="Enter Minute"
              disabled={configLoading}
            />
            {configError && (
              <p className="mt-2 text-xs text-red-400">{configError}</p>
            )}
          </div>

          <div className="rounded-2xl border border-dark-border/60 bg-dark-card/70 p-5 shadow-lg">
            <div className="flex items-center gap-3 mb-4">
              <div className="h-10 w-10 rounded-full bg-[#1f2937] flex items-center justify-center text-green-200">🗣️</div>
              <div>
                <p className="text-sm font-semibold text-white">Language Selection</p>
                <p className="text-xs text-dark-text-muted">Choose the narration language</p>
              </div>
            </div>

            <label className="text-xs font-semibold uppercase tracking-wide text-dark-text-muted">Select Language</label>
            <select
              className="mt-2 w-full rounded-xl border border-dark-border/60 bg-dark-card/70 px-4 py-3 text-sm text-dark-text-primary focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
              value={language}
              onChange={(event) => setLanguage(event.target.value)}
              disabled={configLoading}
            >
              <option value="" disabled>
                {configLoading ? 'Loading options…' : 'Select Language'}
              </option>
              {lectureConfig.languages.map((lang) => (
                <option key={lang.value} value={lang.value}>
                  {lang.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <motion.div
          className="card p-6"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
        >
          <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-6">
            <div>
              <h2 className="text-lg font-semibold">{material.subject}</h2>
              <p className="text-sm text-dark-text-muted">
                {material.board} • Class {material.std}{' '}
                {material.chapter_number ? `• Chapter ${material.chapter_number}` : ''}
              </p>
              {material.author && (
                <p className="text-sm text-dark-text-muted mt-1">Author: {material.author}</p>
              )}
            </div>
            <div className="bg-primary-500/10 border border-primary-500/30 rounded-lg px-4 py-2 text-sm text-primary-200">
              <span className="font-semibold">
                {selectedTopics.length} topic{selectedTopics.length !== 1 ? 's' : ''} selected
              </span>
            </div>
          </div>

          {!lecture && (
            <>
              <div className="mt-6">
                <motion.div
                  className="rounded-lg border border-primary-500/25 bg-primary-500/10 p-5"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25 }}
                >
                  <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                    <div className="space-y-1">
                      <p className="text-sm font-semibold text-primary-100">
                        {material.subject} lecture is ready to generate
                      </p>
                      <p className="text-xs text-primary-100/80">
                        We detected {selectedTopics.length} curated topic{selectedTopics.length !== 1 ? 's' : ''}. Click the button below to generate the full lecture deck in the required format.
                      </p>
                    </div>
                    <div className="rounded-full border border-primary-400/40 bg-black/10 px-4 py-1 text-xs uppercase tracking-wide text-primary-100/90">
                      {selectedTopics.length} selected
                    </div>
                  </div>
                </motion.div>
                {selectedTopics.length === 0 && (
                  <div className="mt-3 rounded-lg border border-dashed border-dark-border px-4 py-3 text-center text-sm text-dark-text-muted">
                    No topics were included in this request. Use the back button to choose at least one topic before generating the lecture.
                  </div>
                )}
              </div>

              <div className="mt-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div className="text-sm text-dark-text-muted flex items-center gap-2">
                  <BookOpen className="w-4 h-4" />
                  You can refine the selection by going back to the materials page.
                </div>
                <Button
                  variant="primary"
                  size="large"
                  icon={Sparkles}
                  onClick={handleBeginGeneration}
                  disabled={selectedTopics.length === 0 || isGenerating}
                  className="flex items-center gap-2"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    'Generate Lecture Content'
                  )}
                </Button>
              </div>
            </>
          )}
        </motion.div>

        {lecture && (
          (() => {
            const slides = Array.isArray(lecture.slides) ? lecture.slides : [];
            const slideCount = slides.length;
            const activeSlide = slides[activeSlideIndex] || null;
            const slideNumber = activeSlide?.number ?? activeSlideIndex + 1;
            const slideMeta = getSlideMeta(slideNumber);
            const slideTheme = slideThemeConfig[slideNumber] || {
              borderClass: 'border-dark-border/60 bg-black/85',
              chipClass: 'text-dark-text-muted bg-dark-border/30',
              headerBadge: 'bg-primary-500 text-white',
              description: `Slide ${slideNumber}`,
            };
            const slideBullets = Array.isArray(activeSlide?.bullets)
              ? activeSlide.bullets.filter(Boolean)
              : [];

            return (
              <>
                <motion.div
                  className="rounded-2xl border border-dark-border/60 bg-dark-card/70 p-6 shadow-lg shadow-black/20"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                    <div className="space-y-4">
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary-200/70">
                          Summary
                        </p>
                        <h2 className="mt-2 text-2xl font-bold text-white">{lecture.title}</h2>
                        <div className="mt-3 flex flex-wrap gap-2 text-xs text-dark-text-muted">
                          {lecture.language && (
                            <span className="rounded-full bg-dark-border/40 px-3 py-1">Language: {lecture.language}</span>
                          )}
                          {lecture.style && (
                            <span className="rounded-full bg-dark-border/40 px-3 py-1">Style: {lecture.style}</span>
                          )}
                          {lecture.estimated_duration && (
                            <span className="rounded-full bg-dark-border/40 px-3 py-1">
                              Duration: {lecture.estimated_duration} mins
                            </span>
                          )}
                          <span className="rounded-full bg-dark-border/40 px-3 py-1">Slides: {slideCount}</span>
                        </div>
                      </div>

                      <div className="rounded-2xl border border-primary-500/30 bg-primary-500/10 p-5 text-sm text-primary-100">
                        {lecture.summary ? (
                          <div className="space-y-3">
                            {renderNarration(
                              lecture.summary,
                              'text-sm leading-relaxed text-primary-100/90',
                            )}
                          </div>
                        ) : (
                          <p className="italic text-primary-100/70">Summary not available for this lecture.</p>
                        )}
                      </div>
                    </div>

                    <div className="flex w-full flex-col gap-3 sm:flex-row sm:items-start sm:justify-end lg:w-auto">
                      <Button
                        variant="secondary"
                        size="small"
                        icon={Clipboard}
                        onClick={handleCopySummary}
                      >
                        Copy Summary
                      </Button>
                      <Button
                        variant="secondary"
                        size="small"
                        icon={Save}
                        onClick={handleSaveLectureDetails}
                      >
                        Save Lecture Details
                      </Button>
                      <Button
                        variant="secondary"
                        size="small"
                        icon={FileDown}
                        onClick={handleDownloadLectureJson}
                      >
                        Download JSON
                      </Button>
                    </div>
                  </div>
                </motion.div>

                <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,260px)_minmax(0,1fr)_minmax(0,320px)]">
                  <section className="rounded-2xl border border-dark-border/60 bg-dark-card/80 p-4 shadow-md">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-semibold text-white">Slides</h3>
                      <span className="text-xs uppercase tracking-wide text-dark-text-muted">
                        Select to edit
                      </span>
                    </div>
                    <div className="mt-4 space-y-2 overflow-y-auto max-h-[520px] pr-1">
                      {slides.length === 0 && (
                        <p className="text-sm text-dark-text-muted">No slides were generated.</p>
                      )}
                      {slides.map((slide, index) => {
                        const number = slide.number ?? index + 1;
                        const meta = getSlideMeta(number);
                        const isActive = index === activeSlideIndex;
                        return (
                          <button
                            key={number}
                            type="button"
                            onClick={() => goToSlide(index)}
                            className={`w-full rounded-xl border px-3 py-3 text-left transition-all ${
                              isActive
                                ? 'border-primary-500 bg-primary-500/20 shadow-lg shadow-primary-500/20'
                                : 'border-dark-border/60 bg-dark-card/60 hover:border-primary-500/40 hover:bg-dark-card/80'
                            }`}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex items-center gap-2">
                                <span
                                  className={`flex h-8 w-8 items-center justify-center rounded-lg text-sm font-semibold ${
                                    isActive ? 'bg-primary-500 text-white' : 'bg-dark-border/60 text-dark-text-muted'
                                  }`}
                                >
                                  {number}
                                </span>
                                <div>
                                  <p className="text-sm font-semibold text-white/90">
                                    {slide.title || meta.heading}
                                  </p>
                                  <p className="text-xs text-dark-text-muted">{meta.description}</p>
                                </div>
                              </div>
                              <span className="text-[10px] font-semibold uppercase tracking-wide text-dark-text-muted">
                                {slideBullets.length ? `${slideBullets.length} pts` : 'Narration'}
                              </span>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </section>

                  <section className="space-y-4 rounded-2xl border border-dark-border/60 bg-dark-card/80 p-6 shadow-md">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.3em] text-dark-text-muted">
                          Preview
                        </p>
                        <h3 className="mt-2 text-lg font-semibold text-white">{slideMeta.heading}</h3>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="secondary"
                          size="small"
                          icon={ChevronLeft}
                          onClick={() => goToSlide(activeSlideIndex - 1)}
                          disabled={activeSlideIndex === 0}
                        >
                          Previous
                        </Button>
                        <span className="text-xs font-semibold uppercase tracking-wide text-dark-text-muted">
                          Slide {activeSlideIndex + 1} of {slideCount}
                        </span>
                        <Button
                          variant="secondary"
                          size="small"
                          icon={ChevronRight}
                          iconPosition="right"
                          onClick={() => goToSlide(activeSlideIndex + 1)}
                          disabled={activeSlideIndex === slideCount - 1}
                        >
                          Next
                        </Button>
                      </div>
                    </div>

                    <div
                      className={`rounded-2xl border-2 px-5 py-6 shadow-lg shadow-black/10 ${slideTheme.borderClass}`}
                    >
                      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                        <span
                          className={`flex h-10 w-10 items-center justify-center rounded-full text-base font-semibold ${slideTheme.headerBadge}`}
                        >
                          {slideNumber}
                        </span>
                        <span
                          className={`rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-wide ${slideTheme.chipClass}`}
                        >
                          {slideTheme.description}
                        </span>
                      </div>
                      <div className="mt-5 space-y-4 text-sm text-white">
                        {slideMeta.description && (
                          <p className="text-xs uppercase tracking-wide text-white/60">
                            {slideMeta.description}
                          </p>
                        )}

                        {renderNarration(activeSlide?.narration)}

                        {slideBullets.length > 0 && (
                          <div>
                            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-dark-text-muted">
                              Bullet Points
                            </h4>
                            <ul className="list-disc space-y-2 pl-6 text-sm text-white/90">
                              {slideBullets.map((bullet, idx) => (
                                <li key={idx}>{bullet}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {activeSlide?.question && (
                          <div className="rounded-xl border border-primary-400/30 bg-primary-500/10 px-4 py-3 text-sm text-primary-100">
                            <p className="text-xs font-semibold uppercase tracking-wide text-primary-100/80">
                              Check Understanding
                            </p>
                            <p className="mt-1 whitespace-pre-line text-primary-50/90">
                              {activeSlide.question}
                            </p>
                          </div>
                        )}
                      </div>

                      <div className="mt-6 flex flex-wrap justify-between gap-3 border-t border-dark-border/40 pt-4">
                        <Button
                          variant="secondary"
                          size="small"
                          onClick={handleScrollToEdit}
                        >
                          Manually Edit
                        </Button>
                        <Button variant="primary" size="small" onClick={handleFinalizeLecture}>
                          Final Generate
                        </Button>
                      </div>
                    </div>
                  </section>

                  <section
                    ref={editPanelRef}
                    className="rounded-2xl border border-dark-border/60 bg-dark-card/80 p-6 shadow-md"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.3em] text-dark-text-muted">
                          Edit Slide
                        </p>
                      </div>
                    </div>

                    <div className="mt-6 space-y-5">
                      <div>
                        <label className="mb-2 block text-xs font-semibold uppercase tracking-wide text-dark-text-muted">
                          Narration
                        </label>
                        <textarea
                          rows={10}
                          value={editedNarration}
                          onChange={(event) => setEditedNarration(event.target.value)}
                          className="w-full rounded-xl border border-dark-border/60 bg-dark-card/70 px-4 py-3 text-sm text-dark-text-primary focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                          placeholder="Use this area to refine the narration for this slide."
                        />
                      </div>

                      <div>
                        <label className="mb-2 block text-xs font-semibold uppercase tracking-wide text-dark-text-muted">
                          Bullets (one per line)
                        </label>
                        <textarea
                          rows={6}
                          value={editedBullets}
                          onChange={(event) => setEditedBullets(event.target.value)}
                          className="w-full rounded-xl border border-dark-border/60 bg-dark-card/70 px-4 py-3 text-sm text-dark-text-primary focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                          placeholder="Add bullet points, each on its own line."
                        />
                      </div>

                      <div className="flex flex-wrap gap-3">
                        <Button variant="primary" size="small" onClick={handleApplySlideEdits}>
                          Apply Edits
                        </Button>
                        <Button variant="secondary" size="small" onClick={handleRestoreSlide}>
                          Restore Slide
                        </Button>
                      </div>
                    </div>

                    <p className="mt-6 text-xs text-dark-text-muted">
                      Tip: Save lecture details once you are satisfied. You can always regenerate later with updated topics for a fresh outline.
                    </p>
                  </section>
                </div>
              </>
            );
          })()
        )}
      </main>
    </div>
  );
};

export default GenerateLecture;
