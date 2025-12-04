import React, { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, Clock, Settings, BookOpen, FileText } from 'lucide-react';
import Button from '../../components/ui/Button';
import toast from 'react-hot-toast';

const RESOLUTION_OPTIONS = ['720p', '1080p', 'HD Output', '5K Output'];

const EditChapter = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const chapter = location.state?.chapter;

  useEffect(() => {
    if (!chapter) {
      toast.error('Select a chapter to edit first.');
      navigate('/chapter-dashboard', { replace: true });
    }
  }, [chapter, navigate]);

  const resolvedChapterTitle = useMemo(() => {
    if (!chapter) return '';
    return chapter.chapter_title || chapter.chapter || chapter.subject || '';
  }, [chapter]);

  const [durationMinutes, setDurationMinutes] = useState(() => chapter?.video?.duration || '');
  const [resolution, setResolution] = useState(() => chapter?.video?.resolution || RESOLUTION_OPTIONS[0]);
  const [chapterName, setChapterName] = useState(() => resolvedChapterTitle);
  const [topicName, setTopicName] = useState('');

  if (!chapter) {
    return null;
  }

  const handleSave = (event) => {
    event.preventDefault();
    // Placeholder save handler – integrate with backend update endpoint when available
    toast.success('Changes saved (sample UI).');
  };

  return (
    <div className="min-h-screen bg-dark-bg text-dark-text-primary p-6">
      <div className="mb-6 flex items-center gap-4">
        <Button
          variant="ghost"
          size="small"
          icon={ArrowLeft}
          onClick={() => navigate(-1)}
        >
          Back
        </Button>
        <div>
          <p className="text-sm uppercase tracking-wide text-dark-text-muted">Edit Chapter</p>
          <h1 className="text-2xl font-bold">{chapter.subject || 'Chapter'}</h1>
          <p className="text-sm text-dark-text-muted">{resolvedChapterTitle}</p>
        </div>
      </div>

      <form onSubmit={handleSave} className="space-y-5">
        {/* Edit Video */}
        <section className="rounded-2xl border border-dark-border bg-dark-card/70 p-5 shadow-lg">
          <div className="mb-4 flex items-center gap-3">
            <Clock className="h-5 w-5 text-primary-300" />
            <div>
              <h2 className="text-lg font-semibold">Edit Video</h2>
              <p className="text-sm text-dark-text-muted">Change Duration</p>
            </div>
          </div>
          <label className="text-xs font-semibold uppercase tracking-wide text-dark-text-muted">Enter Duration in Minutes</label>
          <input
            type="number"
            min={1}
            value={durationMinutes}
            onChange={(event) => setDurationMinutes(event.target.value)}
            className="mt-2 w-full rounded-xl border border-dark-border/60 bg-dark-bg px-4 py-3 text-sm focus:border-primary-500 focus:outline-none"
            placeholder="Duration"
          />
        </section>

        {/* Set Resolution */}
        <section className="rounded-2xl border border-dark-border bg-dark-card/70 p-5 shadow-lg">
          <div className="mb-4 flex items-center gap-3">
            <Settings className="h-5 w-5 text-primary-300" />
            <div>
              <h2 className="text-lg font-semibold">Set Resolution</h2>
              <p className="text-sm text-dark-text-muted">Resolution Option</p>
            </div>
          </div>
          <label className="text-xs font-semibold uppercase tracking-wide text-dark-text-muted">Select Resolution</label>
          <select
            value={resolution}
            onChange={(event) => setResolution(event.target.value)}
            className="mt-2 w-full rounded-xl border border-dark-border/60 bg-dark-bg px-4 py-3 text-sm focus:border-primary-500 focus:outline-none"
          >
            {RESOLUTION_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </section>

        {/* Change Chapter Name */}
        <section className="rounded-2xl border border-dark-border bg-dark-card/70 p-5 shadow-lg">
          <div className="mb-4 flex items-center gap-3">
            <BookOpen className="h-5 w-5 text-primary-300" />
            <div>
              <h2 className="text-lg font-semibold">Change Chapter Name</h2>
              <p className="text-sm text-dark-text-muted">New Chapter Name</p>
            </div>
          </div>
          <label className="text-xs font-semibold uppercase tracking-wide text-dark-text-muted">Enter New Chapter Name</label>
          <input
            type="text"
            value={chapterName}
            onChange={(event) => setChapterName(event.target.value)}
            className="mt-2 w-full rounded-xl border border-dark-border/60 bg-dark-bg px-4 py-3 text-sm focus:border-primary-500 focus:outline-none"
            placeholder="New Chapter Name"
          />
        </section>

        {/* Change Topic Name */}
        <section className="rounded-2xl border border-dark-border bg-dark-card/70 p-5 shadow-lg">
          <div className="mb-4 flex items-center gap-3">
            <FileText className="h-5 w-5 text-primary-300" />
            <div>
              <h2 className="text-lg font-semibold">Change Topic Name</h2>
              <p className="text-sm text-dark-text-muted">New Topic Name</p>
            </div>
          </div>
          <label className="text-xs font-semibold uppercase tracking-wide text-dark-text-muted">Enter New Topic Name</label>
          <input
            type="text"
            value={topicName}
            onChange={(event) => setTopicName(event.target.value)}
            className="mt-2 w-full rounded-xl border border-dark-border/60 bg-dark-bg px-4 py-3 text-sm focus:border-primary-500 focus:outline-none"
            placeholder="New Topic Name"
          />
        </section>

        <div className="flex items-center justify-end gap-3 pt-2">
          <Button variant="ghost" onClick={() => navigate(-1)}>
            Cancel
          </Button>
          <Button type="submit" variant="primary">
            Save Changes
          </Button>
        </div>
      </form>
    </div>
  );
};

export default EditChapter;
