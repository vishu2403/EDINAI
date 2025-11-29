import React, { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { useLocation, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Download,
  Edit3,
  FileText,
  Loader2,
  MessageSquare,
  RefreshCw,
  Save,
  Send,
  Sparkles,
} from 'lucide-react';

import Button from '../../components/ui/Button';
import { chapterMaterialsAPI } from '../../utils/api';

const LectureEditor = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { lecture: incomingLecture, material } = location.state || {};

  const [originalLecture] = useState(() =>
    incomingLecture ? JSON.parse(JSON.stringify(incomingLecture)) : null,
  );
  const [lecture, setLecture] = useState(() =>
    incomingLecture ? JSON.parse(JSON.stringify(incomingLecture)) : null,
  );
  const [activeSlideIndex, setActiveSlideIndex] = useState(0);
  const [narrationDraft, setNarrationDraft] = useState('');
  const [bulletsDraft, setBulletsDraft] = useState('');
  const [titleDraft, setTitleDraft] = useState(incomingLecture?.title || '');
  const [summaryDraft, setSummaryDraft] = useState(incomingLecture?.summary || '');
  const [chatHistoryBySlide, setChatHistoryBySlide] = useState({});
  const [aiPrompt, setAiPrompt] = useState('');
  const [aiLoading, setAiLoading] = useState(false);

  useEffect(() => {
    if (!incomingLecture) {
      toast.error('Lecture data not found. Please generate a lecture first.');
      navigate('/selected-materials', { replace: true });
    }
  }, [incomingLecture, navigate]);

  const slides = lecture?.slides ?? [];
  const activeSlide = slides[activeSlideIndex];
  const activeSlideNumber = activeSlide?.number ?? activeSlideIndex + 1;
  const chatHistory = chatHistoryBySlide[activeSlideNumber] || [];

  useEffect(() => {
    if (!activeSlide) {
      setNarrationDraft('');
      setBulletsDraft('');
      return;
    }
    setNarrationDraft(activeSlide.narration || '');
    setBulletsDraft((activeSlide.bullets || []).join('\n'));
  }, [activeSlide, activeSlideIndex]);

  useEffect(() => {
    setAiPrompt('');
  }, [activeSlideNumber]);

  const originalSlide = useMemo(() => originalLecture?.slides?.[activeSlideIndex], [
    originalLecture,
    activeSlideIndex,
  ]);

  const handleSaveDetails = () => {
    if (!lecture) return;
    setLecture((prev) => ({
      ...prev,
      title: titleDraft.trim() || prev.title,
      summary: summaryDraft,
    }));
    toast.success('Lecture details saved');
  };

  const handleApplyEdits = () => {
    if (!lecture || activeSlideIndex < 0 || activeSlideIndex >= slides.length) {
      return;
    }

    const cleanedBullets = bulletsDraft
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    setLecture((prev) => {
      if (!prev) return prev;
      const updatedSlides = prev.slides.map((slide, index) => {
        if (index !== activeSlideIndex) return slide;
        return {
          ...slide,
          narration: narrationDraft,
          bullets: cleanedBullets,
        };
      });
      return { ...prev, slides: updatedSlides };
    });

    toast.success('Slide updated');
  };

  const handleRestoreSlide = () => {
    if (!originalSlide || !lecture) return;
    setNarrationDraft(originalSlide.narration || '');
    setBulletsDraft((originalSlide.bullets || []).join('\n'));
    setLecture((prev) => {
      if (!prev) return prev;
      const updatedSlides = prev.slides.map((slide, index) =>
        index === activeSlideIndex ? { ...originalSlide } : slide,
      );
      return { ...prev, slides: updatedSlides };
    });
    toast.success('Slide restored');
  };

  const handleRestoreLecture = () => {
    if (!originalLecture) return;
    const restored = JSON.parse(JSON.stringify(originalLecture));
    setLecture(restored);
    setTitleDraft(restored.title || '');
    setSummaryDraft(restored.summary || '');
    setActiveSlideIndex(0);
    toast.success('Lecture restored to original');
  };

  const handleDownloadLecture = () => {
    if (!lecture) return;
    const payload = {
      ...lecture,
      title: titleDraft.trim() || lecture.title,
      summary: summaryDraft,
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${payload.title?.replace(/\s+/g, '_') || 'lecture'}_edited.json`;
    anchor.click();
    window.URL.revokeObjectURL(url);
    toast.success('Edited lecture downloaded');
  };

  const handleSlideChange = (direction) => {
    if (!slides.length) return;
    setActiveSlideIndex((prev) => {
      const next = prev + direction;
      if (next < 0) return 0;
      if (next >= slides.length) return slides.length - 1;
      return next;
    });
  };

  const appendChatMessage = (slideNum, message) => {
    setChatHistoryBySlide((prev) => {
      const existing = prev[slideNum] || [];
      const entry = {
        id: `${slideNum}-${Date.now()}-${existing.length}`,
        ...message,
      };
      return {
        ...prev,
        [slideNum]: [...existing, entry],
      };
    });
  };

  const handleSendAiPrompt = async (event) => {
    event.preventDefault();
    const trimmed = aiPrompt.trim();
    if (!trimmed) {
      toast.error('Instruction લખો પહેલા.');
      return;
    }

    if (!lecture?.lecture_id) {
      toast.error('Lecture ID મળ્યું નથી.');
      return;
    }

    if (!activeSlide) {
      toast.error('કોઈ slide select નથી.');
      return;
    }

    appendChatMessage(activeSlideNumber, { role: 'user', content: trimmed });
    setAiPrompt('');
    setAiLoading(true);

    try {
      const response = await chapterMaterialsAPI.editLectureSlide({
        lectureId: lecture.lecture_id,
        slideNumber: activeSlideNumber,
        instruction: trimmed,
      });

      if (!response.status) {
        throw new Error(response.message || 'Slide edit નિષ્ફળ ગયું.');
      }

      const updatedLecture = response.data?.lecture;
      const updatedSlide = response.data?.slide;
      const assistantMessage = response.message || 'Slide update થઈ ગયું.';

      if (updatedLecture) {
        setLecture(updatedLecture);
      }

      if (updatedSlide) {
        setNarrationDraft(updatedSlide.narration || '');
        setBulletsDraft((updatedSlide.bullets || []).join('\n'));
      }

      appendChatMessage(activeSlideNumber, {
        role: 'assistant',
        content: assistantMessage,
      });
      toast.success('AI assistantથી slide update થયું.');
    } catch (error) {
      const detail = error.response?.data?.detail || error.message || 'AI edit કામ નથી કર્યું.';
      appendChatMessage(activeSlideNumber, {
        role: 'assistant',
        content: `⚠️ ${detail}`,
      });
      toast.error(detail);
    } finally {
      setAiLoading(false);
    }
  };

  const emptyState = !lecture;

  return (
    <div className="min-h-screen bg-dark-bg">
      <header className="bg-dark-card border-b border-dark-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="small"
              icon={ArrowLeft}
              onClick={() => navigate(-1)}
            >
              Back
            </Button>
            <div className="w-10 h-10 bg-purple-600 rounded-full flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-dark-text-primary">Lecture Editor</h1>
              {material ? (
                <p className="text-sm text-dark-text-muted">
                  {material.subject} • Class {material.std} • Chapter {material.chapter_number}
                </p>
              ) : (
                <p className="text-sm text-dark-text-muted">Fine-tune generated lecture slides</p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
            {lecture?.lecture_url && (
              <Button
                variant="ghost"
                size="small"
                icon={FileText}
                onClick={() => window.open(lecture.lecture_url, '_blank')}
              >
                Original JSON
              </Button>
            )}
            <Button
              variant="secondary"
              size="small"
              icon={RefreshCw}
              onClick={handleRestoreLecture}
              disabled={emptyState}
            >
              Restore All
            </Button>
            <Button
              variant="primary"
              size="small"
              icon={Download}
              onClick={handleDownloadLecture}
              disabled={emptyState}
            >
              Download Edited
            </Button>
          </div>
        </div>
      </header>

      {emptyState ? (
        <div className="min-h-[60vh] flex items-center justify-center">
          <div className="text-center space-y-4">
            <BookOpen className="w-12 h-12 text-dark-text-muted mx-auto" />
            <h2 className="text-lg font-semibold text-dark-text-primary">No lecture loaded</h2>
            <p className="text-sm text-dark-text-muted">
              Generate a lecture from the Selected Materials page to start editing.
            </p>
            <Button variant="primary" onClick={() => navigate('/selected-materials')}>
              Go to Selected Materials
            </Button>
          </div>
        </div>
      ) : (
        <main className="p-6 space-y-6">
          {/* Lecture meta */}
          <motion.div
            className="card p-6"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 space-y-3">
                <div>
                  <label className="text-xs uppercase tracking-wide text-dark-text-muted">Lecture Title</label>
                  <input
                    value={titleDraft}
                    onChange={(e) => setTitleDraft(e.target.value)}
                    className="mt-1 w-full bg-dark-bg/60 border border-dark-border rounded-lg px-4 py-2 text-sm text-dark-text-primary focus:border-primary-400 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="text-xs uppercase tracking-wide text-dark-text-muted">Summary</label>
                  <textarea
                    value={summaryDraft}
                    onChange={(e) => setSummaryDraft(e.target.value)}
                    rows={3}
                    className="mt-1 w-full bg-dark-bg/60 border border-dark-border rounded-lg px-4 py-2 text-sm text-dark-text-primary focus:border-primary-400 focus:outline-none"
                  />
                </div>
                <div className="flex flex-wrap gap-3 text-xs text-dark-text-muted">
                  <span>Language: {lecture.language}</span>
                  <span>Style: {lecture.style}</span>
                  <span>Duration: {lecture.estimated_duration} mins</span>
                  <span>Slides: {slides.length}</span>
                </div>
              </div>
              <div className="flex flex-col gap-3 justify-end">
                <Button
                  variant="primary"
                  size="small"
                  icon={Save}
                  onClick={handleSaveDetails}
                >
                  Save Lecture Details
                </Button>
                <Button
                  variant="secondary"
                  size="small"
                  onClick={() => navigator.clipboard.writeText(summaryDraft || '')}
                >
                  Copy Summary
                </Button>
              </div>
            </div>
          </motion.div>

          <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
            {/* Slide list */}
            <motion.div
              className="card p-4 space-y-3"
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-dark-text-primary">Slides</h3>
                <span className="text-xs text-dark-text-muted">Select to edit</span>
              </div>
              <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-1">
                {slides.map((slide, index) => (
                  <button
                    key={slide.number ?? index}
                    className={`w-full text-left px-3 py-2 rounded-lg border transition-colors ${
                      index === activeSlideIndex
                        ? 'border-primary-400 bg-primary-500/10 text-primary-100'
                        : 'border-dark-border bg-dark-bg/60 text-dark-text-muted hover:border-primary-400/40'
                    }`}
                    onClick={() => setActiveSlideIndex(index)}
                  >
                    <p className="text-xs uppercase tracking-wide">Slide {slide.number ?? index + 1}</p>
                    <p className="text-sm font-medium truncate">{slide.title}</p>
                  </button>
                ))}
              </div>
            </motion.div>

            {/* Slide preview */}
            <motion.div
              className="card p-6 xl:col-span-2 space-y-4"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase text-dark-text-muted">Preview</p>
                  <h2 className="text-lg font-semibold text-dark-text-primary">{activeSlide?.title}</h2>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="small"
                    icon={ArrowLeft}
                    onClick={() => handleSlideChange(-1)}
                    disabled={activeSlideIndex === 0}
                  />
                  <span className="text-xs text-dark-text-muted">
                    {activeSlideIndex + 1} / {slides.length}
                  </span>
                  <Button
                    variant="ghost"
                    size="small"
                    icon={ArrowRight}
                    onClick={() => handleSlideChange(1)}
                    disabled={activeSlideIndex === slides.length - 1}
                  />
                </div>
              </div>

              <div className="space-y-4">
                {activeSlide?.narration && (
                  <div className="bg-dark-bg/60 border border-dark-border rounded-lg p-4 space-y-2">
                    <p className="text-xs text-primary-200 uppercase tracking-wide">Narration</p>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap text-dark-text-primary">
                      {activeSlide.narration}
                    </p>
                  </div>
                )}
                {activeSlide?.bullets && activeSlide.bullets.length > 0 && (
                  <div className="bg-dark-bg/60 border border-dark-border rounded-lg p-4 space-y-2">
                    <p className="text-xs text-primary-200 uppercase tracking-wide">Bullets</p>
                    <ul className="list-disc list-outside pl-5 text-sm text-dark-text-primary space-y-1">
                      {activeSlide.bullets.map((bullet, index) => (
                        <li key={index}>{bullet}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {(!activeSlide?.narration || activeSlide.narration.trim().length === 0) &&
                  (!activeSlide?.bullets || activeSlide.bullets.length === 0) && (
                    <div className="bg-dark-bg/40 border border-dashed border-dark-border rounded-lg p-6 text-center text-sm text-dark-text-muted">
                      This slide does not contain content yet. Use the editor to add narration or bullets.
                    </div>
                  )}
              </div>
            </motion.div>

            {/* Slide editor */}
            <div className="xl:col-span-1 space-y-6">
              <motion.div
                className="card p-6 space-y-4"
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
              >
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-dark-text-primary">Edit Slide</h3>
                  <span className="text-xs text-dark-text-muted">Manual adjustments</span>
                </div>

                <div className="space-y-3">
                  <label className="text-xs uppercase tracking-wide text-dark-text-muted">Narration</label>
                  <textarea
                    value={narrationDraft}
                    onChange={(e) => setNarrationDraft(e.target.value)}
                    rows={8}
                    className="w-full bg-dark-bg/60 border border-dark-border rounded-lg px-4 py-3 text-sm text-dark-text-primary focus:border-primary-400 focus:outline-none"
                    placeholder="Write or edit narration here"
                  />
                </div>

                <div className="space-y-3">
                  <label className="text-xs uppercase tracking-wide text-dark-text-muted">Bullets (one per line)</label>
                  <textarea
                    value={bulletsDraft}
                    onChange={(e) => setBulletsDraft(e.target.value)}
                    rows={6}
                    className="w-full bg-dark-bg/60 border border-dark-border rounded-lg px-4 py-3 text-sm text-dark-text-primary focus:border-primary-400 focus:outline-none"
                    placeholder="Key points, each on a new line"
                  />
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <Button
                    variant="primary"
                    size="small"
                    icon={Edit3}
                    onClick={handleApplyEdits}
                  >
                    Apply Edits
                  </Button>
                  <Button
                    variant="secondary"
                    size="small"
                    icon={RefreshCw}
                    onClick={handleRestoreSlide}
                    disabled={!originalSlide}
                  >
                    Restore Slide
                  </Button>
                  <Button
                    variant="ghost"
                    size="small"
                    icon={FileText}
                    onClick={() => navigator.clipboard.writeText(narrationDraft)}
                    disabled={!narrationDraft}
                  >
                    Copy Narration
                  </Button>
                </div>
              </motion.div>

              {/* AI assistant */}
              <motion.div
                className="card p-6 space-y-4"
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="w-4 h-4 text-primary-300" />
                    <h3 className="text-sm font-semibold text-dark-text-primary">AI Chatbot</h3>
                  </div>
                  <span className="text-xs text-dark-text-muted">Slide {activeSlideNumber}</span>
                </div>

                <p className="text-xs text-dark-text-muted">
                  Slide માટે instruction લખો. ઉદાહરણ: "Narration ને વધુ સરળ બનાવો" અથવા "Add 3 bullet examples".
                </p>

                <div className="bg-dark-bg/60 border border-dark-border rounded-lg p-3 space-y-3 max-h-64 overflow-y-auto scrollbar-thin scrollbar-thumb-primary-400/40">
                  {chatHistory.length === 0 ? (
                    <div className="text-xs text-dark-text-muted text-center">
                      હજુ સુધી chat નથી. Instruction મોકલો અને AI update કરશે.
                    </div>
                  ) : (
                    chatHistory.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.role === 'assistant' ? 'justify-start' : 'justify-end'}`}
                      >
                        <div
                          className={`px-3 py-2 rounded-lg text-sm max-w-[85%] whitespace-pre-wrap ${
                            message.role === 'assistant'
                              ? 'bg-primary-500/10 text-primary-100 border border-primary-500/30'
                              : 'bg-dark-card text-dark-text-primary border border-dark-border'
                          }`}
                        >
                          {message.content}
                        </div>
                      </div>
                    ))
                  )}
                </div>

                <form onSubmit={handleSendAiPrompt} className="space-y-3">
                  <textarea
                    value={aiPrompt}
                    onChange={(e) => setAiPrompt(e.target.value)}
                    rows={3}
                    className="w-full bg-dark-bg/60 border border-dark-border rounded-lg px-3 py-2 text-sm text-dark-text-primary focus:border-primary-400 focus:outline-none"
                    placeholder="AI ને કહો કે શું બદલવું છે..."
                    disabled={aiLoading}
                  />
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-dark-text-muted">
                      AI language: {lecture?.language || 'English'} • Style: {lecture?.style || 'storytelling'}
                    </span>
                    <Button
                      type="submit"
                      variant="primary"
                      size="small"
                      icon={aiLoading ? Loader2 : Send}
                      iconPosition="right"
                      loading={aiLoading}
                      disabled={aiLoading}
                    >
                      {aiLoading ? 'Processing...' : 'Send Prompt'}
                    </Button>
                  </div>
                </form>
              </motion.div>
            </div>
          </div>

          {/* Quiz section */}
          <motion.div
            className="card p-6 space-y-4"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-dark-text-primary">Quiz & Reflection</h3>
              <span className="text-xs text-dark-text-muted">Edit questions inline</span>
            </div>

            <div className="space-y-3">
              {lecture.quiz_questions?.map((question, index) => (
                <div key={index} className="flex items-start gap-3">
                  <span className="text-xs text-primary-300 mt-1">Q{index + 1}</span>
                  <textarea
                    value={question}
                    onChange={(e) =>
                      setLecture((prev) => {
                        if (!prev) return prev;
                        const updated = [...prev.quiz_questions];
                        updated[index] = e.target.value;
                        return { ...prev, quiz_questions: updated };
                      })
                    }
                    rows={2}
                    className="flex-1 bg-dark-bg/60 border border-dark-border rounded-lg px-3 py-2 text-sm text-dark-text-primary focus:border-primary-400 focus:outline-none"
                  />
                </div>
              ))}
            </div>
          </motion.div>
        </main>
      )}
    </div>
  );
};

export default LectureEditor;
