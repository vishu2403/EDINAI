import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Home,
  BookOpen,
  GraduationCap,
  Hash,
  Layers,
  Mail,
  MessageCircle,
  PlayCircle,
  ClipboardList,
  ShoppingBag,
  User,
  Users,
  LogOut,
  Search,
  Moon,
  Bell,
  Play,
  Phone,
  PhoneCall,
  Loader2,
  Languages,
  Lock,
  Settings,
  Heart,
  HeartOff,
  MessageSquare,
  Share2,
  Bookmark,
  Clock,
  Calendar,
  CheckCircle2,
  Flame,
  ChevronRight,
  ChevronDown,
  RefreshCcw,
  Trash2,
  ExternalLink,
  Check,
} from 'lucide-react';

import toast from 'react-hot-toast';

import { schoolPortalAPI, getFileUrl } from '../../utils/api';
import StudentPortalCall from '../../components/StudentPortalCall';

const navItems = [
  { label: 'Home', icon: Home },
  { label: 'Book', icon: BookOpen },
  { label: 'Chat', icon: MessageCircle },
  { label: 'Watched Lecture', icon: PlayCircle },
  { label: 'Saved Video', icon: ClipboardList },
  { label: 'Purchase History', icon: ShoppingBag },
  { label: 'Settings', icon: Settings },
  { label: 'Profile', icon: User },
];

const PROFILE_DATA_KEY = 'inai_student_profile_data';
const PROFILE_ENROLLMENT_KEY = 'inai_student_profile_enrollment';
const PROFILE_COMPLETE_KEY = 'inai_student_profile_complete';
const SETTINGS_LANGUAGE_KEY = 'inai_student_language_preference';
const SETTINGS_NOTIFICATION_KEY = 'inai_student_notification_settings';

const fallbackProfile = {
  firstName: 'Rahul Sharma',
  fatherName: 'Rahul Sharma',
  classStream: '12th Science',
  division: 'A',
  classHead: 'Mrs. Priya Verma',
  enrollmentNumber: 'ENR2024001234',
  mobileNumber: '15486 58962',
  parentsNumber: '25847 36958',
  email: 'rahul.sharma@student.edu',
  advisorEmail: 'priya.verma@inai.edu',
  photoDataUrl:
    'https://images.unsplash.com/photo-1524504388940-b1c1722653e1?auto=format&fit=facearea&w=256&h=256&q=80',
};

const formatDurationLabel = (seconds) => {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return '—';
  }
  const totalSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = totalSeconds % 60;
  return `${minutes}:${String(remainingSeconds).padStart(2, '0')}`;
};

const getRelativeDateLabel = (isoDate) => {
  if (!isoDate) {
    return 'Recently added';
  }
  const date = new Date(isoDate);

  if (Number.isNaN(date.getTime())) {
    return 'Recently added';
  }
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  if (diffDays <= 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  return date.toLocaleDateString();
};

const formatUploadSubtitle = (isoDate) => {
  if (!isoDate) {
    return 'Uploaded recently';
  }
  try {
    const date = new Date(isoDate);
    if (Number.isNaN(date.getTime())) {
      return 'Uploaded recently';
    }
    return `Uploaded on ${date.toLocaleDateString('en-IN', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
    })}`;
  } catch {
    return `Uploaded on ${isoDate}`;
  }
};

const DEFAULT_SUBJECT_FILTERS = ['All subject', 'Physics', 'Mathematics', 'Biology', 'English'];
const LANGUAGE_OPTIONS = ['English', 'Gujarati', 'Hindi'];
const NOTIFICATION_OPTIONS = [
  {
    key: 'email',
    label: 'Email Notification',
    description: 'Receive notifications via email',
  },
  {
    key: 'push',
    label: 'Push Notification',
    description: 'Receive push notifications on your device',
  },
  {
    key: 'sms',
    label: 'SMS Notification',
    description: 'Receive notifications via SMS',
  },
];

const COMMENT_VIEWS = {
  VIDEO: 'video',
  ALL: 'all',
};

const STUDENT_TOKEN_KEY = 'inai_student_token';

const getInitials = (name) =>
  name
    .split(' ')
    .map((part) => part[0] || '')
    .join('')
    .slice(0, 2)
    .toUpperCase();

const isYouTubeUrl = (value) => {
  if (!value) return false;
  try {
    const url = new URL(value);
    return url.hostname.includes('youtube.com') || url.hostname.includes('youtu.be');
  } catch {
    return false;
  }
};

const getYouTubeEmbedUrl = (value) => {
  if (!isYouTubeUrl(value)) return null;
  try {
    const url = new URL(value);
    let videoId = '';
    if (url.hostname.includes('youtu.be')) {
      videoId = url.pathname.replace('/', '');
    } else if (url.searchParams.has('v')) {
      videoId = url.searchParams.get('v');
    } else if (url.pathname.startsWith('/embed/')) {
      videoId = url.pathname.replace('/embed/', '');
    }

    return videoId ? `https://www.youtube.com/embed/${videoId}` : null;
  } catch {
    return null;
  }
};

const LINK_EXTRACT_REGEX = /(https?:\/\/[^\s]+)/i;
const VIDEO_FILE_REGEX = /(\.(mp4|m4v|mov|webm|ogg|ogv|avi|mkv))(?:$|\?)/i;

const sanitizeSharedUrl = (value) => value.replace(/[),.;]+$/, '');

const extractFirstUrl = (text) => {
  if (!text) return null;
  const match = text.match(LINK_EXTRACT_REGEX);
  if (!match) {
    return null;
  }
  return sanitizeSharedUrl(match[0]);
};

const isVideoFileUrl = (value) => {
  if (!value) return false;
  try {
    const parsed = new URL(value);
    return VIDEO_FILE_REGEX.test(parsed.pathname.toLowerCase());
  } catch {
    return VIDEO_FILE_REGEX.test(value.toLowerCase());
  }
};

const splitMessageLines = (text) =>
  (text || '')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length);

const normalizeShareMetadata = (metadata) => {
  if (!metadata) return null;
  if (typeof metadata === 'string') {
    try {
      return JSON.parse(metadata);
    } catch {
      return null;
    }
  }
  if (typeof metadata === 'object') {
    return metadata;
  }
  return null;
};

const buildPreviewFromMetadata = (rawMetadata) => {
  const metadata = normalizeShareMetadata(rawMetadata);
  if (!metadata?.url) {
    return null;
  }
  const url = sanitizeSharedUrl(metadata.url);
  return {
    url,
    title: metadata.title || 'Shared resource',
    subject: metadata.subject || null,
    description: metadata.description || metadata.summary || null,
    thumbnail: metadata.thumbnail_url || metadata.thumbnailUrl || null,
    isYouTube: isYouTubeUrl(url),
    isVideoFile: isVideoFileUrl(url),
  };
};

const buildPreviewFromMessage = (text) => {
  const url = extractFirstUrl(text);
  if (!url) {
    return null;
  }
  const lines = splitMessageLines(text);
  const titleCandidate = lines[0]?.includes('http') ? 'Shared link' : lines[0] || 'Shared link';
  const subjectLine = lines.find((line) => /^subject:/i.test(line));
  const descriptionLine = lines.find(
    (line) => line !== titleCandidate && line !== subjectLine && !line.includes(url),
  );

  return {
    url,
    title: titleCandidate,
    subject: subjectLine ? subjectLine.replace(/^subject:\s*/i, '').trim() : null,
    description: descriptionLine || null,
    isYouTube: isYouTubeUrl(url),
    isVideoFile: isVideoFileUrl(url),
  };
};

const getChatSharePreview = (message) =>
  buildPreviewFromMetadata(message?.share_metadata) ?? buildPreviewFromMessage(message?.message);

const StudentPortalDashboard = () => {
  const navigate = useNavigate();
  const [activeNav, setActiveNav] = useState('Home');
  const [selectedChatEnrollment, setSelectedChatEnrollment] = useState(null);
  const [chatSearchTerm, setChatSearchTerm] = useState('');
  const [bookSearchTerm, setBookSearchTerm] = useState('');
  const [selectedSubject, setSelectedSubject] = useState('All subject');
  const [profileData, setProfileData] = useState(fallbackProfile);
  const [languagePreference, setLanguagePreference] = useState('English');
  const [notificationSettings, setNotificationSettings] = useState({
    email: true,
    push: true,
    sms: true,
  });
  const [passwordForm, setPasswordForm] = useState({
    current: '',
    next: '',
    confirm: '',
  });
  const [isPasswordUpdating, setIsPasswordUpdating] = useState(false);
  const [isCallActive, setIsCallActive] = useState(false);
  const [peers, setPeers] = useState([]);
  const [messages, setMessages] = useState({});
  const [chatLoading, setChatLoading] = useState(false);
  const [peersLoading, setPeersLoading] = useState(false);
  const [messageDraft, setMessageDraft] = useState('');
  const [chatError, setChatError] = useState(null);
  const [books, setBooks] = useState([]);
  const [booksLoading, setBooksLoading] = useState(false);
  const [booksError, setBooksError] = useState(null);
  const [videos, setVideos] = useState([]);
  const [selectedVideoId, setSelectedVideoId] = useState(null);
  const [videoDetail, setVideoDetail] = useState(null);
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoError, setVideoError] = useState(null);
  const [relatedVideos, setRelatedVideos] = useState([]);
  const [comments, setComments] = useState([]);
  const [commentsOpen, setCommentsOpen] = useState(false);
  const [commentsView, setCommentsView] = useState(COMMENT_VIEWS.VIDEO);
  const [commentDraft, setCommentDraft] = useState('');
  const [isSubmittingComment, setIsSubmittingComment] = useState(false);
  const [isSharingVideo, setIsSharingVideo] = useState(false);
  const [isShareDialogOpen, setIsShareDialogOpen] = useState(false);
  const [shareSelectedRecipients, setShareSelectedRecipients] = useState([]);
  const [shareSearchTerm, setShareSearchTerm] = useState('');
  const [allComments, setAllComments] = useState([]);
  const [allCommentsLoading, setAllCommentsLoading] = useState(false);
  const [allCommentsError, setAllCommentsError] = useState(null);
  const [watchedSummary, setWatchedSummary] = useState({
    watched_videos: 0,
    completed_videos: 0,
    total_watch_seconds: 0,
    total_records: 0,
  });
  const [watchedVideos, setWatchedVideos] = useState([]);
  const [watchedLoading, setWatchedLoading] = useState(false);
  const [watchedError, setWatchedError] = useState(null);
  const [savedVideos, setSavedVideos] = useState([]);
  const [savedLoading, setSavedLoading] = useState(false);
  const [savedError, setSavedError] = useState(null);
  const commentListRef = useRef(null);
  const savedVideoCount = savedVideos.length;
  const isAllCommentsView = commentsView === COMMENT_VIEWS.ALL;
  const drawerComments = isAllCommentsView ? allComments : comments;
  const canViewVideoComments = Boolean(selectedVideoId);

  const callClientId = useMemo(() => {
    const enrollment = (profileData?.enrollmentNumber ?? '').trim();
    if (enrollment) {
      return enrollment.replace(/\s+/g, '-');
    }

    const email = (profileData?.email ?? '').trim();
    if (email) {
      return email;
    }

    return `${PROFILE_DATA_KEY}-guest`;
  }, [profileData]);

  const handleLogout = () => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(PROFILE_COMPLETE_KEY, 'true');
    }

    setProfileData(fallbackProfile);
    setSelectedChatEnrollment(null);
    setActiveNav('Home');
    setSelectedSubject('All subject');
    setBookSearchTerm('');
    navigate('/school-portal');
  };

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const storedLanguage = window.localStorage.getItem(SETTINGS_LANGUAGE_KEY);
    if (storedLanguage) {
      setLanguagePreference(storedLanguage);
    }

    const storedNotifications = window.localStorage.getItem(SETTINGS_NOTIFICATION_KEY);
    if (storedNotifications) {
      try {
        const parsed = JSON.parse(storedNotifications);
        setNotificationSettings((previous) => ({
          email: typeof parsed.email === 'boolean' ? parsed.email : previous.email,
          push: typeof parsed.push === 'boolean' ? parsed.push : previous.push,
          sms: typeof parsed.sms === 'boolean' ? parsed.sms : previous.sms,
        }));
      } catch {
        // Ignore parsing errors
      }
    }
  }, []);

  const fetchSavedVideos = useCallback(async (showSpinner = true) => {
    if (showSpinner) {
      setSavedLoading(true);
    }
    setSavedError(null);
    try {
      const response = await schoolPortalAPI.getSavedVideos();
      if (!response.status) {
        throw new Error(response.message || 'Unable to load saved videos');
      }
      const data = response.data ?? {};
      setSavedVideos(data.videos ?? []);
    } catch (error) {
      const message = error.response?.data?.detail || error.message || 'Unable to load saved videos';
      setSavedError(message);
      toast.error(message);
    } finally {
      setSavedLoading(false);
    }
  }, []);

  const handleLanguageChange = (event) => {
    const value = event.target.value;
    setLanguagePreference(value);
    if (typeof window !== 'undefined') {
      try {
        window.localStorage.setItem(SETTINGS_LANGUAGE_KEY, value);
      } catch {
        // Ignore storage errors
      }
    }
    toast.success(`Language updated to ${value}`);
  };

  const handlePasswordInputChange = (field) => (event) => {
    const { value } = event.target;
    setPasswordForm((previous) => ({
      ...previous,
      [field]: value,
    }));
  };

  const handlePasswordSubmit = async (event) => {
    event.preventDefault();

    const trimmedCurrent = passwordForm.current.trim();
    const trimmedNext = passwordForm.next.trim();
    const trimmedConfirm = passwordForm.confirm.trim();

    if (!trimmedCurrent || !trimmedNext || !trimmedConfirm) {
      toast.error('Please fill in all password fields.');
      return;
    }

    if (trimmedNext !== trimmedConfirm) {
      toast.error('New password and confirmation do not match.');
      return;
    }

    try {
      setIsPasswordUpdating(true);
      const response = await schoolPortalAPI.changePassword(trimmedCurrent, trimmedNext);
      if (!response?.status) {
        const message = response?.message || 'Unable to update password right now.';
        toast.error(message);
        return;
      }

      toast.success(response.message || 'Password updated successfully.');
      setPasswordForm({ current: '', next: '', confirm: '' });
    } catch (error) {
      const message =
        error.response?.data?.detail || error.response?.data?.message || error.message || 'Unable to update password.';
      toast.error(message);
    } finally {
      setIsPasswordUpdating(false);
    }
  };

  const toggleNotificationSetting = (key) => {
    setNotificationSettings((previous) => {
      const nextValue = !previous[key];
      const updated = {
        ...previous,
        [key]: nextValue,
      };

      if (typeof window !== 'undefined') {
        try {
          window.localStorage.setItem(SETTINGS_NOTIFICATION_KEY, JSON.stringify(updated));
        } catch {
          // Ignore storage errors
        }
      }

      toast.success(`${nextValue ? 'Enabled' : 'Disabled'} ${key} notifications.`);
      return updated;
    });
  };

  const handleNavChange = (label) => {
    setActiveNav(label);
    if (label === 'Chat') {
      setChatSearchTerm('');
    }
    if (label === 'Book') {
      setSelectedSubject('All subject');
      setBookSearchTerm('');
    }
  };

  useEffect(() => {
    if (typeof window === 'undefined') return;

    let isMounted = true;

    const transformProfile = (profile) => {
      if (!profile) return null;
      return {
        firstName: profile.first_name ?? profile.firstName ?? '',
        fatherName: profile.father_name ?? profile.fatherName ?? '',
        classStream: profile.class_stream ?? profile.classStream ?? '',
        division: profile.division ?? profile.division ?? '',
        classHead: profile.class_head ?? profile.classHead ?? '',
        enrollmentNumber: profile.enrollment_number ?? profile.enrollmentNumber ?? '',
        mobileNumber: profile.mobile_number ?? profile.mobileNumber ?? '',
        parentsNumber: profile.parents_number ?? profile.parentsNumber ?? '',
        email: profile.email ?? profile.email ?? '',
        advisorEmail: profile.advisorEmail ?? null,
        photoDataUrl: profile.photo_path ? getFileUrl(profile.photo_path) : profile.photoDataUrl ?? null,
      };
    };

    const persistProfile = (data, enrollment) => {
      try {
        const payload = {
          ...data,
          advisorEmail: data.advisorEmail ?? null,
          photoDataUrl: data.photoDataUrl ?? null,
        };
        window.localStorage.setItem(PROFILE_DATA_KEY, JSON.stringify(payload));
        if (enrollment) {
          window.localStorage.setItem(PROFILE_ENROLLMENT_KEY, enrollment);
        }
      } catch {
        // Ignore storage errors (e.g., quota exceeded)
      }
    };

    const loadProfile = async () => {
      const storedValue = window.localStorage.getItem(PROFILE_DATA_KEY);
      const storedEnrollment = window.localStorage.getItem(PROFILE_ENROLLMENT_KEY);
      let parsed = null;

      if (storedValue) {
        try {
          parsed = JSON.parse(storedValue);
          if (isMounted) {
            setProfileData({
              ...fallbackProfile,
              ...parsed,
              photoDataUrl: parsed?.photoDataUrl ?? fallbackProfile.photoDataUrl,
            });
          }
        } catch {
          if (isMounted) {
            setProfileData(fallbackProfile);
          }
        }
      } else if (isMounted) {
        setProfileData(fallbackProfile);
      }

      const enrollmentToFetch = (storedEnrollment || parsed?.enrollmentNumber || parsed?.enrollment_number)?.trim();
      if (!enrollmentToFetch) {
        return;
      }

      try {
        const response = await schoolPortalAPI.getProfile(enrollmentToFetch);
        const transformed = transformProfile(response);
        if (transformed && isMounted) {
          const merged = {
            ...fallbackProfile,
            ...transformed,
          };
          setProfileData({
            ...merged,
            photoDataUrl: transformed.photoDataUrl ?? fallbackProfile.photoDataUrl,
          });
          persistProfile(merged, transformed.enrollmentNumber);
        }
      } catch (error) {
        if (error?.response?.status === 404) {
          // Clear stale enrollment reference
          window.localStorage.removeItem(PROFILE_ENROLLMENT_KEY);
        }
      }
    };

    loadProfile();

    const handleStorage = (event) => {
      if ([PROFILE_DATA_KEY, PROFILE_ENROLLMENT_KEY].includes(event.key)) {
        loadProfile();
      }
    };

    window.addEventListener('storage', handleStorage);
    return () => {
      isMounted = false;
      window.removeEventListener('storage', handleStorage);
    };
  }, []);

  const filteredPeers = peers.filter((peer) =>
    peer.name?.toLowerCase().includes(chatSearchTerm.toLowerCase())
  );

  const shareFilteredPeers = useMemo(() => {
    const term = shareSearchTerm.trim().toLowerCase();
    if (!term) {
      return peers;
    }
    return peers.filter((peer) => {
      const name = peer.name?.toLowerCase() ?? '';
      const enrollment = peer.enrollment_number?.toLowerCase() ?? '';
      return name.includes(term) || enrollment.includes(term);
    });
  }, [peers, shareSearchTerm]);

  const resolvedClassStream = profileData.classStream?.trim() || fallbackProfile.classStream;

  const subjectFilters = useMemo(() => {
    const uniqueSubjects = Array.from(
      new Set(books.map((book) => book.subject).filter(Boolean)),
    );
    if (!uniqueSubjects.length) {
      return DEFAULT_SUBJECT_FILTERS;
    }
    return ['All subject', ...uniqueSubjects];
  }, [books]);

  useEffect(() => {
    if (selectedSubject !== 'All subject' && !subjectFilters.includes(selectedSubject)) {
      setSelectedSubject('All subject');
    }
  }, [selectedSubject, subjectFilters]);

  const fetchDashboardVideos = useCallback(async () => {
    setVideoLoading(true);
    setVideoError(null);
    try {
      const response = await schoolPortalAPI.getDashboardVideos(6);
      if (!response.status) {
        throw new Error(response.message || 'Unable to load videos');
      }
      const payload = response.data?.videos ?? [];
      setVideos(payload);
      if (payload.length) {
        setSelectedVideoId((previous) => previous ?? payload[0].id ?? null);
      } else {
        setVideoDetail(null);
        setSelectedVideoId(null);
      }
    } catch (error) {
      const message = error.response?.data?.detail || error.message || 'Unable to load videos';
      setVideoError(message);
      toast.error(message);
    } finally {
      setVideoLoading(false);
    }
  }, []);

  const fetchVideoDetail = useCallback(
    async (videoId) => {
      if (!videoId) return;
      setVideoLoading(true);
      setVideoError(null);
      try {
        const response = await schoolPortalAPI.getVideoDetail(videoId);
        if (!response.status) {
          throw new Error(response.message || 'Unable to load lecture');
        }
        const detail = response.data?.video ?? null;
        const fetchedComments = response.data?.comments ?? [];
        setVideoDetail(detail);
        setComments(fetchedComments);
      } catch (error) {
        const message = error.response?.data?.detail || error.message || 'Unable to load lecture';
        setVideoError(message);
        toast.error(message);
      } finally {
        setVideoLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (selectedVideoId) {
      fetchVideoDetail(selectedVideoId);
    }
  }, [selectedVideoId, fetchVideoDetail]);

  useEffect(() => {
    if (!videos.length) {
      setRelatedVideos([]);
      return;
    }
    const remaining = videos.filter((video) => video.id !== selectedVideoId);
    setRelatedVideos(remaining.slice(0, 4));
  }, [videos, selectedVideoId]);

  const refreshComments = useCallback(async (videoId) => {
    if (!videoId) return;
    try {
      const response = await schoolPortalAPI.getVideoComments(videoId);
      if (!response.status) {
        throw new Error(response.message || 'Unable to load comments');
      }
      setComments(response.data?.comments ?? []);
    } catch (error) {
      const message = error.response?.data?.detail || error.message || 'Unable to load comments';
      toast.error(message);
    }
  }, []);

  const fetchAllComments = useCallback(async () => {
    setAllCommentsLoading(true);
    setAllCommentsError(null);
    try {
      const response = await schoolPortalAPI.getAllVideoComments();
      if (!response.status) {
        throw new Error(response.message || 'Unable to load comments');
      }
      setAllComments(response.data?.comments ?? []);
    } catch (error) {
      const message = error.response?.data?.detail || error.message || 'Unable to load comments';
      setAllCommentsError(message);
      toast.error(message);
    } finally {
      setAllCommentsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (commentsOpen && isAllCommentsView) {
      fetchAllComments();
    }
  }, [commentsOpen, isAllCommentsView, fetchAllComments]);

  const handleCommentsViewChange = useCallback(
    (nextView) => {
      if (nextView === commentsView) {
        return;
      }
      setCommentsView(nextView);
      if (nextView === COMMENT_VIEWS.ALL) {
        fetchAllComments();
      } else if (selectedVideoId) {
        refreshComments(selectedVideoId);
      }
    },
    [commentsView, fetchAllComments, refreshComments, selectedVideoId],
  );

  const handleOpenVideoComments = useCallback(() => {
    if (selectedVideoId) {
      refreshComments(selectedVideoId);
    }
    setCommentsView(COMMENT_VIEWS.VIDEO);
    setCommentsOpen(true);
  }, [selectedVideoId, refreshComments]);

  const handleOpenAllComments = useCallback(() => {
    setCommentsView(COMMENT_VIEWS.ALL);
    setCommentsOpen(true);
    fetchAllComments();
  }, [fetchAllComments]);

  const handleSubmitComment = useCallback(
    async (event) => {
      event?.preventDefault();
      if (!selectedVideoId) return;
      const trimmed = commentDraft.trim();
      if (!trimmed) {
        toast.error('Comment cannot be empty.');
        return;
      }

      setIsSubmittingComment(true);
      try {
        const response = await schoolPortalAPI.addVideoComment(selectedVideoId, trimmed);
        if (!response.status) {
          throw new Error(response.message || 'Unable to post comment');
        }
        setCommentDraft('');
        setCommentsView(COMMENT_VIEWS.VIDEO);
        await refreshComments(selectedVideoId);
        setCommentsOpen(true);
        requestAnimationFrame(() => {
          if (commentListRef.current) {
            commentListRef.current.scrollTop = 0;
          }
        });
      } catch (error) {
        const message = error.response?.data?.detail || error.message || 'Unable to post comment';
        toast.error(message);
      } finally {
        setIsSubmittingComment(false);
      }
    },
    [selectedVideoId, commentDraft, refreshComments],
  );

  const toggleVideoLike = useCallback(async () => {
    if (!selectedVideoId || !videoDetail) return;
    const nextLiked = !videoDetail.user_liked;
    setVideoDetail((previous) =>
      previous
        ? {
            ...previous,
            user_liked: nextLiked,
            total_likes: Math.max(0, (previous.total_likes ?? 0) + (nextLiked ? 1 : -1)),
          }
        : previous,
    );
    try {
      await schoolPortalAPI.setVideoLike(selectedVideoId, nextLiked);
    } catch (error) {
      setVideoDetail((previous) =>
        previous
          ? {
              ...previous,
              user_liked: !nextLiked,
              total_likes: Math.max(0, (previous.total_likes ?? 0) + (nextLiked ? -1 : 1)),
            }
          : previous,
      );
      const message = error.response?.data?.detail || error.message || 'Unable to update like';
      toast.error(message);
    }
  }, [selectedVideoId, videoDetail]);

  const toggleVideoSubscription = useCallback(async () => {
    if (!selectedVideoId || !videoDetail) return;
    const nextSubscribed = !videoDetail.user_subscribed;
    setVideoDetail((previous) =>
      previous
        ? {
            ...previous,
            user_subscribed: nextSubscribed,
            total_subscribers: Math.max(0, (previous.total_subscribers ?? 0) + (nextSubscribed ? 1 : -1)),
          }
        : previous,
    );
    try {
      await schoolPortalAPI.setVideoSubscribe(selectedVideoId, nextSubscribed);
      fetchSavedVideos(false);
    } catch (error) {
      setVideoDetail((previous) =>
        previous
          ? {
              ...previous,
              user_subscribed: !nextSubscribed,
              total_subscribers: Math.max(0, (previous.total_subscribers ?? 0) + (nextSubscribed ? -1 : 1)),
            }
          : previous,
      );
      const message = error.response?.data?.detail || error.message || 'Unable to update subscription';
      toast.error(message);
    }
  }, [selectedVideoId, videoDetail, fetchSavedVideos]);

  const recordVideoProgress = useCallback(
    async (seconds) => {
      if (!selectedVideoId || seconds <= 0) return;
      try {
        await schoolPortalAPI.recordVideoWatch(selectedVideoId, seconds);
      } catch {
        // Ignore non-critical error
      }
    },
    [selectedVideoId],
  );

  const fetchWatchedLectures = useCallback(async () => {
    setWatchedLoading(true);
    setWatchedError(null);
    try {
      const response = await schoolPortalAPI.getWatchedLectures();
      if (!response.status) {
        throw new Error(response.message || 'Unable to load watch history');
      }
      const data = response.data ?? {};
      setWatchedSummary(data.summary ?? {
        watched_videos: 0,
        completed_videos: 0,
        total_watch_seconds: 0,
        total_records: 0,
      });
      setWatchedVideos(data.videos ?? []);
    } catch (error) {
      const message = error.response?.data?.detail || error.message || 'Unable to load watch history';
      setWatchedError(message);
    } finally {
      setWatchedLoading(false);
    }
  }, []);

  const subjectFilteredBooks = selectedSubject === 'All subject'
    ? books
    : books.filter((book) => book.subject === selectedSubject);
  const displayBooks = subjectFilteredBooks.filter((book) =>
    book.title.toLowerCase().includes(bookSearchTerm.toLowerCase())
  );

  const activeConversation = peers.find((peer) => peer.enrollment_number === selectedChatEnrollment) ?? null;
  const activeMessages = selectedChatEnrollment ? messages[selectedChatEnrollment] || [] : [];
  const callRoomId = activeConversation ? `chat-${activeConversation.enrollment_number}` : null;

  useEffect(() => {
    if (!activeConversation) {
      setIsCallActive(false);
    }
  }, [activeConversation]);

  const refreshPeers = useCallback(async () => {
    setPeersLoading(true);
    setChatError(null);
    try {
      const response = await schoolPortalAPI.getChatPeers();
      if (!response.status) {
        throw new Error(response.message || 'Unable to load classmates');
      }
      const nextPeers = response.data?.peers ?? [];
      setPeers(nextPeers);
      if (nextPeers.length && !selectedChatEnrollment) {
        setSelectedChatEnrollment(nextPeers[0].enrollment_number);
      }
    } catch (error) {
      const message = error.response?.data?.detail || error.message || 'Unable to load classmates';
      setChatError(message);
      toast.error(message);
    } finally {
      setPeersLoading(false);
    }
  }, [selectedChatEnrollment]);

  const fetchMessages = useCallback(
    async (peerEnrollment) => {
      if (!peerEnrollment) return;
      setChatLoading(true);
      setChatError(null);
      try {
        const response = await schoolPortalAPI.getChatMessages(peerEnrollment);
        if (!response.status) {
          throw new Error(response.message || 'Unable to load messages');
        }
        setMessages((prev) => ({
          ...prev,
          [peerEnrollment]: response.data?.messages ?? [],
        }));
      } catch (error) {
        const message = error.response?.data?.detail || error.message || 'Unable to load messages';
        setChatError(message);
        toast.error(message);
      } finally {
        setChatLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (activeNav === 'Chat') {
      refreshPeers();
    }
    if (activeNav === 'Home') {
      fetchDashboardVideos();
      fetchWatchedLectures();
    }
    if (activeNav === 'Watched Lecture') {
      fetchWatchedLectures();
    }
    if (activeNav === 'Saved Video') {
      fetchSavedVideos();
    }
  }, [activeNav, refreshPeers, fetchDashboardVideos, fetchWatchedLectures, fetchSavedVideos]);

  useEffect(() => {
    if (selectedChatEnrollment && activeNav === 'Chat' && !messages[selectedChatEnrollment]) {
      fetchMessages(selectedChatEnrollment);
    }
  }, [activeNav, selectedChatEnrollment, messages, fetchMessages]);

  const handleSelectPeer = (enrollment) => {
    setSelectedChatEnrollment(enrollment);
    if (!messages[enrollment]) {
      fetchMessages(enrollment);
    }
  };

  const handleSendMessage = async (event) => {
    event.preventDefault();
    const trimmed = messageDraft.trim();
    if (!trimmed || !selectedChatEnrollment) {
      return;
    }

    try {
      const response = await schoolPortalAPI.sendChatMessage(selectedChatEnrollment, trimmed);
      if (!response.status) {
        throw new Error(response.message || 'Unable to send message');
      }
      const updatedMessages = response.data?.messages ?? [];
      setMessages((prev) => ({
        ...prev,
        [selectedChatEnrollment]: updatedMessages,
      }));
      setMessageDraft('');
    } catch (error) {
      const message = error.response?.data?.detail || error.message || 'Unable to send message';
      toast.error(message);
    }
  };

  const handleOpenShareDialog = useCallback(() => {
    if (!videoDetail) {
      toast.error('Select a lecture before sharing.');
      return;
    }
    setShareSearchTerm('');
    setShareSelectedRecipients((previous) => {
      if (previous.length) {
        return previous;
      }
      if (selectedChatEnrollment) {
        return [selectedChatEnrollment];
      }
      return [];
    });
    setIsShareDialogOpen(true);
  }, [videoDetail, selectedChatEnrollment]);

  const handleCloseShareDialog = useCallback(() => {
    setIsShareDialogOpen(false);
  }, []);

  const handleToggleShareRecipient = useCallback((enrollment) => {
    setShareSelectedRecipients((previous) => {
      if (previous.includes(enrollment)) {
        return previous.filter((value) => value !== enrollment);
      }
      return [...previous, enrollment];
    });
  }, []);

  const shareVideoToSelectedPeers = useCallback(async () => {
    if (!videoDetail) {
      toast.error('Select a lecture before sharing.');
      return;
    }

    const recipients = shareSelectedRecipients.length
      ? shareSelectedRecipients
      : selectedChatEnrollment
        ? [selectedChatEnrollment]
        : [];
    const uniqueRecipients = Array.from(new Set(recipients));

    if (!uniqueRecipients.length) {
      toast.error('Select at least one student to share this video.');
      return;
    }

    const shareLines = [videoDetail.title || 'Lecture'];
    if (videoDetail.subject) {
      shareLines.push(`Subject: ${videoDetail.subject}`);
    }
    if (videoDetail.video_url) {
      shareLines.push(videoDetail.video_url);
    }

    const payload = shareLines.join('\n');
    const shareMetadata = videoDetail.video_url
      ? {
          type: videoDetail.video_type || 'lecture',
          title: videoDetail.title || 'Lecture',
          subject: videoDetail.subject || null,
          description: videoDetail.description || null,
          url: sanitizeSharedUrl(videoDetail.video_url),
          thumbnail_url: videoDetail.thumbnail_url || null,
          duration_seconds: videoDetail.duration_seconds ?? null,
          lecture_id: videoDetail.id ?? null,
          provider: videoDetail.teacher_name || videoDetail.publisher || 'INAI',
        }
      : null;

    try {
      setIsSharingVideo(true);
      for (const enrollment of uniqueRecipients) {
        const response = await schoolPortalAPI.sendChatMessage(enrollment, payload, shareMetadata);
        if (!response.status) {
          throw new Error(response.message || 'Unable to share video');
        }
        const updatedMessages = response.data?.messages ?? [];
        setMessages((prev) => ({
          ...prev,
          [enrollment]: updatedMessages,
        }));
      }
      setShareSelectedRecipients(uniqueRecipients);
      setShareSearchTerm('');
      setSelectedChatEnrollment(uniqueRecipients[0] ?? null);
      setActiveNav('Chat');
      setIsShareDialogOpen(false);
      toast.success(
        uniqueRecipients.length === 1 ? 'Video shared with the student.' : 'Video shared with selected classmates.',
      );
    } catch (error) {
      const message = error.response?.data?.detail || error.message || 'Unable to share video';
      toast.error(message);
    } finally {
      setIsSharingVideo(false);
    }
  }, [videoDetail, shareSelectedRecipients, selectedChatEnrollment]);

  useEffect(() => {
    if (!isShareDialogOpen) {
      return;
    }
    if (peers.length || peersLoading) {
      return;
    }
    refreshPeers();
  }, [isShareDialogOpen, peers.length, peersLoading, refreshPeers]);

  const fetchBooks = useCallback(
    async (subjectFilter) => {
      const subjectParam = subjectFilter && subjectFilter !== 'All subject' ? subjectFilter : undefined;
      setBooksLoading(true);
      setBooksError(null);
      try {
        const response = await schoolPortalAPI.getBooks(subjectParam);
        if (!response.status) {
          throw new Error(response.message || 'Unable to load books');
        }
        setBooks(response.data?.books ?? []);
      } catch (error) {
        const message = error.response?.data?.detail || error.message || 'Unable to load books';
        setBooksError(message);
        toast.error(message);
      } finally {
        setBooksLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (activeNav === 'Book') {
      fetchBooks(selectedSubject);
    }
  }, [activeNav, selectedSubject, fetchBooks]);

  const formatValue = (value, fallbackText = 'Not provided') => {
    if (value === undefined || value === null) return fallbackText;
    const text = String(value).trim();
    return text.length ? text : fallbackText;
  };

  const getProfileValue = (key, fallbackText = 'Not provided') => {
    const candidate = profileData?.[key];
    if (candidate === undefined || candidate === null || String(candidate).trim() === '') {
      return formatValue(fallbackProfile?.[key], fallbackText);
    }
    return formatValue(candidate, fallbackText);
  };

  const resolvedFullName = profileData.firstName?.trim() || fallbackProfile.firstName;
  const resolvedPhoto = profileData.photoDataUrl ?? fallbackProfile.photoDataUrl;
  const displayInitials = getInitials(resolvedFullName);
  const hasPhoto = Boolean(resolvedPhoto && resolvedPhoto.trim() !== '');

  const profileSections = [
    {
      title: 'Personal Information',
      items: [
        { label: 'Full Name', value: resolvedFullName, icon: User },
        { label: 'Father Name', value: getProfileValue('fatherName'), icon: Users },
        { label: 'Enrollment Number', value: getProfileValue('enrollmentNumber'), icon: Hash },
        { label: 'Email Address', value: getProfileValue('email', 'Not linked'), icon: Mail },
      ],
    },
    {
      title: 'Academic Information',
      items: [
        { label: 'Class / Stream', value: getProfileValue('classStream'), icon: GraduationCap },
        { label: 'Division', value: getProfileValue('division'), icon: Layers },
        { label: 'Class Head', value: getProfileValue('classHead', 'Not assigned'), icon: Users },
        { label: 'Advisor Email', value: getProfileValue('advisorEmail', 'Not assigned'), icon: Mail },
      ],
    },
    {
      title: 'Contact Information',
      items: [
        { label: 'Mobile Number', value: getProfileValue('mobileNumber'), icon: Phone },
        { label: 'Parents Number', value: getProfileValue('parentsNumber'), icon: PhoneCall },
      ],
    },
  ];

  const formatWatchTime = useMemo(() => {
    const seconds = watchedSummary.total_watch_seconds ?? 0;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;
    if (hours) {
      return `${hours}h ${minutes}m ${remainingSeconds}s`;
    }
    if (minutes) {
      return `${minutes}m ${remainingSeconds}s`;
    }
    return `${remainingSeconds}s`;
  }, [watchedSummary.total_watch_seconds]);

  const completionPercent = useMemo(() => {
    if (!watchedSummary.total_records) return 0;
    const completed = watchedSummary.completed_videos ?? 0;
    return Math.round((completed / watchedSummary.total_records) * 100);
  }, [watchedSummary.completed_videos, watchedSummary.total_records]);

  const renderVideoCard = (video) => {
    const isActive = video.id === selectedVideoId;
    return (
      <article
        key={video.id}
        className={`group overflow-hidden rounded-3xl border border-white/5 bg-[#141722] shadow-xl transition hover:border-primary-500/40 ${
          isActive ? 'border-primary-500/40 ring-1 ring-primary-500/40' : ''
        }`}
      >
        <button
          type="button"
          onClick={() => setSelectedVideoId(video.id)}
          className="relative block w-full text-left"
        >
          <img
            src={video.thumbnail_url || fallbackProfile.photoDataUrl}
            alt={video.title || 'Lecture thumbnail'}
            className="h-48 w-full object-cover transition duration-500 group-hover:scale-105"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent" />
          <div className="absolute bottom-4 left-4 right-4 flex items-start gap-4">
            <span className="inline-flex h-12 w-12 items-center justify-center rounded-full bg-white text-black shadow-lg">
              <Play className="h-4 w-4" />
            </span>
            <div className="space-y-1">
              <p className="text-xs uppercase tracking-[0.4em] text-white/40">{video.subject || 'Lecture'}</p>
              <p className="text-sm font-semibold text-white line-clamp-2">{video.title || 'Untitled Lecture'}</p>
            </div>
          </div>
          <span className="absolute right-4 top-4 rounded-full bg-black/60 px-3 py-1 text-xs text-white">
            {video.duration_seconds ? `${Math.floor(video.duration_seconds / 60)}:${String(video.duration_seconds % 60).padStart(2, '0')}` : '—'}
          </span>
        </button>
        <div className="space-y-3 px-5 py-4 text-sm text-white/80">
          <p className="line-clamp-2 text-white/70">
            {video.description || 'Continue exploring this lecture to strengthen your understanding.'}
          </p>
          <div className="flex items-center justify-between text-xs text-white/40">
            <span className="inline-flex items-center gap-1">
              <Heart className="h-3.5 w-3.5" /> {(video.total_likes ?? 0).toLocaleString('en-IN')}
            </span>
            <span className="inline-flex items-center gap-1">
              <MessageSquare className="h-3.5 w-3.5" /> {(video.total_comments ?? 0).toLocaleString('en-IN')}
            </span>
            <span className="inline-flex items-center gap-1">
              <Users className="h-3.5 w-3.5" />
              {(video.total_watch_count ?? 0).toLocaleString('en-IN')} viewers
            </span>
          </div>
          <p className="text-[11px] text-white/40">{formatUploadSubtitle(video.created_at)}</p>
        </div>
      </article>
    );
  };

  useEffect(() => {
    fetchSavedVideos(false);
  }, [fetchSavedVideos]);

  const handleRemoveSavedVideo = useCallback(
    async (videoId) => {
      if (!videoId) return;
      try {
        await schoolPortalAPI.setVideoSubscribe(videoId, false);
        if (videoDetail?.id === videoId) {
          setVideoDetail((previous) =>
            previous
              ? {
                  ...previous,
                  user_subscribed: false,
                }
              : previous,
          );
        }
        await fetchSavedVideos(false);
        toast.success('Removed from saved videos');
      } catch (error) {
        const message = error.response?.data?.detail || error.message || 'Unable to remove from saved';
        toast.error(message);
      }
    },
    [fetchSavedVideos, videoDetail],
  );

  return (
    <div className="min-h-screen bg-[#0E1018] text-white">
      <div className="flex min-h-screen">
        {/* Sidebar */}
        <aside className="hidden w-64 flex-col justify-between border-r border-white/5 bg-[#11131D] px-6 py-10 lg:flex">
          <div className="space-y-10">
            <div>
              <div className="text-4xl font-black tracking-[0.5em]">INAI</div>
              <div className="mt-2 text-xs uppercase tracking-[0.6em] text-white/50">Verse</div>
            </div>

            <nav className="space-y-2">
              {navItems.map(({ label, icon: Icon }) => (
                <button
                  key={label}
                  type="button"
                  onClick={() => handleNavChange(label)}
                  className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm transition ${
                    activeNav === label
                      ? 'bg-primary-500/10 text-primary-200'
                      : 'text-white/70 hover:bg-white/5 hover:text-white'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </button>
              ))}
            </nav>
          </div>

          <button
            type="button"
            onClick={handleLogout}
            className="flex items-center gap-3 rounded-xl px-4 py-3 text-sm text-white/60 transition hover:bg-white/5"
          >
            <LogOut className="h-4 w-4" />
            Logout
          </button>
        </aside>

        {/* Main content */}
        <main className="flex flex-1 flex-col">
          {activeNav === 'Chat' ? (
            <div className="flex flex-1 flex-col">
              <header className="flex items-center justify-between border-b border-white/5 bg-[#10121C] px-6 py-6">
                <div>
                  <p className="text-xs uppercase tracking-[0.4em] text-white/40">chat</p>
                  <h1 className="mt-1 text-2xl font-semibold">Classmates</h1>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    className="rounded-full bg-[#1A1D29] p-2 text-white/60 transition hover:text-white"
                  >
                    <Moon className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    className="rounded-full bg-[#1A1D29] p-2 text-white/60 transition hover:text-white"
                  >
                    <Bell className="h-4 w-4" />
                  </button>
                  <div className="h-9 w-9 rounded-full bg-gradient-to-br from-primary-400 to-primary-600" />
                </div>
              </header>

              <section className="flex-1 overflow-hidden">
                <aside className="hidden w-[24rem] flex-shrink-0 border-r border-white/5 bg-[#10131E] 2xl:block">
                  <div className="space-y-5 px-6 py-6">
                    <div>
                      <p className="text-xs uppercase tracking-[0.4em] text-white/40">Students</p>
                      <h2 className="mt-2 text-lg font-semibold">Same Division</h2>
                    </div>
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-white/40" />
                      <input
                        type="search"
                        value={chatSearchTerm}
                        onChange={(event) => setChatSearchTerm(event.target.value)}
                        placeholder="Search student..."
                        className="w-full rounded-full bg-[#1A1D29] py-2 pl-10 pr-4 text-sm text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-primary-400/40"
                      />
                    </div>
                  </div>

                  <div className="h-full overflow-y-auto px-3 pb-6">
                    {peersLoading ? (
                      <div className="flex items-center justify-center py-10 text-white/70">
                        <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Loading classmates…
                      </div>
                    ) : filteredPeers.length === 0 ? (
                      <div className="mx-3 mt-10 rounded-2xl border border-white/5 bg-[#121520] px-4 py-6 text-center text-sm text-white/50">
                        {chatError || 'No classmates found.'}
                      </div>
                    ) : (
                      filteredPeers.map((peer) => (
                        <button
                          key={peer.enrollment_number}
                          type="button"
                          onClick={() => handleSelectPeer(peer.enrollment_number)}
                          className={`flex w-full items-start gap-4 rounded-2xl px-4 py-4 text-left transition ${
                            selectedChatEnrollment === peer.enrollment_number
                              ? 'bg-primary-500/10 text-primary-100'
                              : 'bg-[#121520] hover:bg-[#191D2A]'
                          }`}
                        >
                          <div className="flex h-11 w-11 items-center justify-center rounded-full bg-gradient-to-br from-primary-400 to-primary-600 text-sm font-semibold">
                            {getInitials(peer.name)}
                          </div>
                          <div className="flex flex-1 flex-col gap-1">
                            <div className="flex items-center justify-between gap-3">
                              <span className="text-sm font-semibold text-white/90">{peer.name}</span>
                              <span className="text-xs text-white/40">
                                {peer.last_message_at ? new Date(peer.last_message_at).toLocaleString() : ''}
                              </span>
                            </div>
                            <p className="truncate text-xs text-white/60">{peer.last_message || 'No messages yet'}</p>
                          </div>
                        </button>
                      ))
                    )}
                  </div>
                </aside>

                <div className="flex flex-1 flex-col bg-[#0F121B]">
                  {activeConversation ? (
                    <>
                      <div className="flex items-center justify-between border-b border-white/5 bg-[#11131F] px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-primary-400 to-primary-600 text-xs font-semibold">
                            {getInitials(activeConversation.name)}
                          </div>
                          <div>
                            <p className="text-sm font-semibold text-white/90">{activeConversation.name}</p>
                            <p className="text-xs text-white/50">Division {activeConversation.division || '—'}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3 text-white/50">
                          <button type="button" className="rounded-full bg-[#1A1D29] px-3 py-1 text-xs transition hover:text-white">
                            View Profile
                          </button>
                          <button
                            type="button"
                            disabled={!callRoomId}
                            onClick={() => setIsCallActive((previous) => !previous)}
                            className={`rounded-full px-3 py-1 text-xs transition ${
                              isCallActive
                                ? 'bg-red-500/20 text-red-200 hover:bg-red-500/30'
                                : 'bg-primary-500/20 text-primary-100 hover:bg-primary-500/30'
                            } ${!callRoomId ? 'cursor-not-allowed opacity-50 hover:bg-primary-500/20' : ''}`}
                          >
                            {isCallActive ? 'Hide Call' : 'Start Call'}
                          </button>
                        </div>
                      </div>

                      <div className="flex-1 space-y-6 overflow-y-auto px-6 py-8">
                        {chatLoading ? (
                          <div className="flex items-center justify-center py-10 text-white/70">
                            <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Loading messages…
                          </div>
                        ) : activeMessages.length ? (
                          activeMessages.map((message) => {
                            const isOutgoing = message.sender_enrollment === profileData.enrollmentNumber;
                            const messageLines = splitMessageLines(message.message);
                            const previewData = getChatSharePreview(message);
                            const bubbleBaseClasses = isOutgoing
                              ? 'bg-primary-500/30 text-primary-50'
                              : 'bg-[#161926] text-white/80';

                            return (
                              <div key={message.id} className={`flex ${isOutgoing ? 'justify-end' : 'justify-start'}`}>
                                <div className={`max-w-[70%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${bubbleBaseClasses}`}>
                                  {messageLines.length ? (
                                    <div className="space-y-1">
                                      {messageLines.map((line, index) => (
                                        <p key={`${message.id}-line-${index}`}>{line}</p>
                                      ))}
                                    </div>
                                  ) : (
                                    <p>{message.message}</p>
                                  )}

                                  {previewData ? (
                                    <div className="mt-3 space-y-3 rounded-2xl border border-white/10 bg-[#0F111C] p-3 text-left text-white">
                                      <div className="space-y-1">
                                        <p className="text-[10px] uppercase tracking-[0.3em] text-white/40">Shared lecture</p>
                                        <p className="text-sm font-semibold text-white/90">{previewData.title}</p>
                                        {previewData.subject ? (
                                          <p className="text-xs text-white/60">Subject · {previewData.subject}</p>
                                        ) : null}
                                        {previewData.description ? (
                                          <p className="text-xs text-white/60">{previewData.description}</p>
                                        ) : null}
                                      </div>

                                      {previewData.isYouTube ? (
                                        <div className="aspect-video w-full overflow-hidden rounded-xl border border-white/5 bg-black/40">
                                          <iframe
                                            title={previewData.title}
                                            src={getYouTubeEmbedUrl(previewData.url) ?? previewData.url}
                                            className="h-full w-full"
                                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                            allowFullScreen
                                          />
                                        </div>
                                      ) : previewData.isVideoFile ? (
                                        <div className="overflow-hidden rounded-xl border border-white/5 bg-black/40">
                                          <video controls className="h-full w-full" preload="metadata">
                                            <source src={previewData.url} />
                                            Your browser does not support the video tag.
                                          </video>
                                        </div>
                                      ) : null}

                                      <a
                                        href={previewData.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-white transition hover:bg-white/20"
                                      >
                                        <ExternalLink className="h-4 w-4" />
                                        {previewData.isYouTube || previewData.isVideoFile ? 'Open full video' : 'Open link'}
                                      </a>
                                    </div>
                                  ) : null}

                                  <span className={`mt-2 block text-[10px] ${isOutgoing ? 'text-primary-100/70' : 'text-white/40'}`}>
                                    {message.created_at ? new Date(message.created_at).toLocaleTimeString() : ''}
                                  </span>
                                </div>
                              </div>
                            );
                          })
                        ) : (
                          <div className="py-12 text-center text-sm text-white/50">No messages yet. Say hello! 😊</div>
                        )}
                      </div>

                      {isCallActive && callRoomId ? (
                        <div className="border-t border-white/5 bg-[#0F121B] px-6 py-6">
                          <StudentPortalCall roomId={callRoomId} clientId={callClientId} onClose={() => setIsCallActive(false)} />
                        </div>
                      ) : null}

                      <div className="border-t border-white/5 bg-[#11131F] px-4 py-4">
                        <form onSubmit={handleSendMessage} className="flex items-center gap-3 rounded-full bg-[#1A1D29] px-4 py-2">
                          <input
                            type="text"
                            value={messageDraft}
                            onChange={(event) => setMessageDraft(event.target.value)}
                            placeholder="Type a message"
                            className="flex-1 bg-transparent text-sm text-white placeholder:text-white/40 focus:outline-none"
                          />
                          <button type="submit" className="rounded-full bg-primary-500 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-black transition hover:bg-primary-400" disabled={!selectedChatEnrollment || !messageDraft.trim()}>
                            Send
                          </button>
                        </form>
                      </div>
                    </>
                  ) : (
                    <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
                      <div className="flex h-20 w-20 items-center justify-center rounded-full border border-dashed border-white/20">
                        <MessageCircle className="h-8 w-8 text-white/30" />
                      </div>
                      <h2 className="mt-6 text-lg font-semibold text-white/90">Select a Student</h2>
                      <p className="mt-2 max-w-sm text-sm text-white/40">
                        Choose a student from the left panel to start chatting.
                      </p>
                    </div>
                  )}
                </div>
              </section>
            </div>
          ) : activeNav === 'Watched Lecture' ? (
            <div className="flex flex-1 flex-col bg-[#0F111B]">
              <header className="flex flex-col gap-6 border-b border-white/5 bg-[#10121C] px-6 py-6 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.4em] text-white/40">{resolvedClassStream}</p>
                  <h1 className="mt-1 text-2xl font-semibold">Watched lectures</h1>
                  <p className="mt-1 text-sm text-white/50">Every lecture you've already watched appears here.</p>
                </div>

                <div className="flex w-full flex-col gap-4 lg:w-auto lg:flex-row lg:items-center lg:justify-end">
                  <button
                    type="button"
                    onClick={fetchWatchedLectures}
                    className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.3em] text-white transition hover:bg-white/20"
                  >
                    Refresh <RefreshCcw className="h-4 w-4" />
                  </button>
                </div>
              </header>

              <section className="flex-1 overflow-y-auto px-6 py-8">
                {watchedLoading ? (
                  <div className="flex h-full min-h-[40vh] items-center justify-center rounded-3xl border border-white/5 bg-[#141722] px-6 py-12 text-white/70">
                    <Loader2 className="mr-2 h-6 w-6 animate-spin" /> Loading watched lectures…
                  </div>
                ) : watchedVideos.length ? (
                  <div className="space-y-4">
                    {watchedVideos.map((video) => {
                      const totalDuration = video.duration_seconds ?? 0;
                      const watchedDuration = video.watch_duration_seconds ?? video.user_watch_duration_seconds ?? 0;
                      const clampedWatch = totalDuration ? Math.min(watchedDuration, totalDuration) : watchedDuration;
                      const completionPercent = totalDuration
                        ? Math.min(100, Math.round((clampedWatch / totalDuration) * 100))
                        : 0;
                      const watchedDateLabel = video.last_watched_at
                        ? new Date(video.last_watched_at).toLocaleDateString()
                        : 'Recently added';

                      return (
                        <article
                          key={video.id}
                          className="flex flex-col gap-4 rounded-3xl border border-white/5 bg-[#141722] px-5 py-5 text-white/80 transition hover:border-primary-500/30 md:flex-row"
                        >
                          <button
                            type="button"
                            onClick={() => {
                              setSelectedVideoId(video.id);
                              setActiveNav('Home');
                            }}
                            className="relative w-full overflow-hidden rounded-2xl border border-white/5 bg-black/30 shadow-inner md:w-72"
                          >
                            <img
                              src={video.thumbnail_url || fallbackProfile.photoDataUrl}
                              alt={video.title || 'Lecture thumbnail'}
                              className="h-48 w-full object-cover"
                            />
                            <div className="absolute inset-x-4 bottom-3 rounded-full bg-black/70 px-3 py-1 text-xs text-white/80">
                              <span className="inline-flex items-center gap-1">
                                <Clock className="h-3.5 w-3.5" />
                                {formatDurationLabel(clampedWatch)} / {formatDurationLabel(totalDuration)}
                              </span>
                            </div>
                            <div className="absolute inset-x-0 bottom-0 h-1 bg-white/10">
                              <div className="h-full bg-primary-500" style={{ width: `${completionPercent}%` }} />
                            </div>
                          </button>

                          <div className="flex flex-1 flex-col gap-3">
                            <div>
                              <p className="text-xs uppercase tracking-[0.4em] text-white/40">{video.subject || 'Lecture'}</p>
                              <h3 className="text-lg font-semibold text-white">{video.title || 'Untitled Lecture'}</h3>
                              <p className="text-sm text-white/60">
                                {video.chapter_name || video.description || 'Keep strengthening this concept.'}
                              </p>
                            </div>
                            <div className="space-y-1 text-xs text-white/60">
                              <p className="inline-flex items-center gap-2">
                                <Calendar className="h-4 w-4" /> Watched Date · {watchedDateLabel}
                              </p>
                              <p className="font-semibold text-white/80">{completionPercent}% Complete</p>
                            </div>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                ) : (
                  <div className="rounded-3xl border border-dashed border-white/10 bg-[#141722] px-6 py-16 text-center text-sm text-white/50">
                    {watchedError || 'Watch your first lecture to see it appear here.'}
                  </div>
                )}
              </section>
            </div>
          ) : activeNav === 'Saved Video' ? (
            <div className="flex flex-1 flex-col bg-[#0F111B]">
              <header className="flex flex-col gap-6 border-b border-white/5 bg-[#10121C] px-6 py-6 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.4em] text-white/40">{resolvedClassStream}</p>
                  <h1 className="mt-1 text-2xl font-semibold">Saved videos</h1>
                  <p className="mt-1 text-sm text-white/50">Revisit lectures you've bookmarked to rewatch later.</p>
                </div>

                <div className="flex w-full flex-col gap-4 lg:w-auto lg:flex-row lg:items-center lg:justify-end">
                  <button
                    type="button"
                    onClick={fetchSavedVideos}
                    className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.3em] text-white transition hover:bg-white/20"
                  >
                    Refresh <RefreshCcw className="h-4 w-4" />
                  </button>
                </div>
              </header>

              <section className="flex-1 overflow-y-auto px-6 py-8">
                {savedLoading ? (
                  <div className="flex h-full min-h-[40vh] items-center justify-center rounded-3xl border border-white/5 bg-[#141722] px-6 py-12 text-white/70">
                    <Loader2 className="mr-2 h-6 w-6 animate-spin" /> Loading saved lectures…
                  </div>
                ) : savedVideos.length ? (
                  <div className="space-y-4">
                    {savedVideos.map((video) => (
                      <article
                        key={video.id}
                        className="flex flex-col gap-4 rounded-2xl border border-white/5 bg-[#141722] px-5 py-4 text-white/80 transition hover:border-primary-500/30 md:flex-row md:items-center"
                      >
                        <div className="flex flex-1 items-start gap-4">
                          <button
                            type="button"
                            onClick={() => {
                              setSelectedVideoId(video.id);
                              setActiveNav('Home');
                            }}
                            className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-xl bg-white/10 text-white transition hover:bg-white/20"
                          >
                            <Play className="h-5 w-5" />
                          </button>
                          <div className="space-y-1">
                            <p className="text-xs uppercase tracking-[0.35em] text-white/40">{video.subject || 'Lecture'}</p>
                            <h3 className="text-base font-semibold text-white">{video.title || 'Untitled Lecture'}</h3>
                            <p className="text-sm text-white/50">
                              {video.chapter_name || video.description || 'Keep learning from this saved session.'}
                            </p>
                          </div>
                        </div>

                        <div className="flex items-center gap-2 self-start md:ml-auto">
                          <button
                            type="button"
                            onClick={() => handleRemoveSavedVideo(video.id)}
                            className="rounded-full p-2 text-white/60 transition hover:bg-white/10 hover:text-white"
                            aria-label="Remove saved video"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </article>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-3xl border border-dashed border-white/10 bg-[#141722] px-6 py-16 text-center text-sm text-white/50">
                    {savedError || 'You have not saved any lectures yet. Subscribe to videos to see them here.'}
                  </div>
                )}
              </section>
            </div>
          ) : activeNav === 'Settings' ? (
            <div className="flex flex-1 flex-col bg-[#0F111B]">
              <header className="flex items-center justify-between border-b border-white/5 bg-[#10121C] px-6 py-6">
                <div>
                  <p className="text-xs uppercase tracking-[0.4em] text-white/40">settings</p>
                  <h1 className="mt-1 text-2xl font-semibold text-white">Personalize your experience</h1>
                  <p className="mt-1 text-sm text-white/50">Choose language, update password, and control notifications.</p>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    className="rounded-full bg-[#1A1D29] p-2 text-white/60 transition hover:text-white"
                  >
                    <Moon className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    className="rounded-full bg-[#1A1D29] p-2 text-white/60 transition hover:text-white"
                  >
                    <Bell className="h-4 w-4" />
                  </button>
                  <div className="h-9 w-9 rounded-full bg-gradient-to-br from-primary-400 to-primary-600" />
                </div>
              </header>

              <section className="flex-1 overflow-y-auto px-6 py-10">
                <div className="mx-auto flex w-full max-w-5xl flex-col gap-10">
                  <section className="rounded-3xl border border-white/5 bg-[#10131F] p-6 shadow-lg sm:p-8">
                    <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                      <div className="flex items-center gap-3">
                        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#1B2031] text-primary-200">
                          <Languages className="h-5 w-5" />
                        </div>
                        <div>
                          <h2 className="text-lg font-semibold text-white">Language Section</h2>
                          <p className="text-sm text-white/50">Choose your preferred language for the interface.</p>
                        </div>
                      </div>
                    </header>
                    <div className="mt-6">
                      <label className="flex flex-col gap-2 text-sm">
                        <span className="text-xs font-semibold uppercase tracking-[0.3em] text-white/40">Language</span>
                        <select
                          value={languagePreference}
                          onChange={handleLanguageChange}
                          className="w-full rounded-xl border border-white/10 bg-[#171B27] px-4 py-3 text-sm text-white shadow-inner focus:outline-none focus:ring-2 focus:ring-primary-400/40"
                        >
                          {LANGUAGE_OPTIONS.map((option) => (
                            <option key={option} value={option} className="bg-[#171B27]">
                              {option}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                  </section>

                  <section className="rounded-3xl border border-white/5 bg-[#10131F] p-6 shadow-lg sm:p-8">
                    <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                      <div className="flex items-center gap-3">
                        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#1B2031] text-primary-200">
                          <Lock className="h-5 w-5" />
                        </div>
                        <div>
                          <h2 className="text-lg font-semibold text-white">Update Password</h2>
                          <p className="text-sm text-white/50">Change your password to keep your account secure.</p>
                        </div>
                      </div>
                    </header>

                    <form onSubmit={handlePasswordSubmit} className="mt-6 space-y-4">
                      <label className="block text-sm">
                        <span className="text-xs font-semibold uppercase tracking-[0.3em] text-white/40">Current Password</span>
                        <input
                          type="password"
                          value={passwordForm.current}
                          onChange={handlePasswordInputChange('current')}
                          placeholder="Enter current password"
                          className="mt-2 w-full rounded-xl border border-white/10 bg-[#171B27] px-4 py-3 text-sm text-white placeholder:text-white/30 focus:outline-none focus:ring-2 focus:ring-primary-400/40"
                        />
                      </label>
                      <label className="block text-sm">
                        <span className="text-xs font-semibold uppercase tracking-[0.3em] text-white/40">New Password</span>
                        <input
                          type="password"
                          value={passwordForm.next}
                          onChange={handlePasswordInputChange('next')}
                          placeholder="Enter new password"
                          className="mt-2 w-full rounded-xl border border-white/10 bg-[#171B27] px-4 py-3 text-sm text-white placeholder:text-white/30 focus:outline-none focus:ring-2 focus:ring-primary-400/40"
                        />
                      </label>
                      <label className="block text-sm">
                        <span className="text-xs font-semibold uppercase tracking-[0.3em] text-white/40">Confirm New Password</span>
                        <input
                          type="password"
                          value={passwordForm.confirm}
                          onChange={handlePasswordInputChange('confirm')}
                          placeholder="Confirm new password"
                          className="mt-2 w-full rounded-xl border border-white/10 bg-[#171B27] px-4 py-3 text-sm text-white placeholder:text-white/30 focus:outline-none focus:ring-2 focus:ring-primary-400/40"
                        />
                      </label>
                      <button
                        type="submit"
                        className="inline-flex w-full items-center justify-center rounded-full bg-primary-500 px-4 py-3 text-xs font-semibold uppercase tracking-[0.3em] text-black transition hover:bg-primary-400 disabled:cursor-not-allowed disabled:opacity-60"
                        disabled={isPasswordUpdating}
                      >
                        {isPasswordUpdating ? 'Updating…' : 'Update Password'}
                      </button>
                    </form>
                  </section>

                  <section className="rounded-3xl border border-white/5 bg-[#10131F] p-6 shadow-lg sm:p-8">
                    <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                      <div className="flex items-center gap-3">
                        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#1B2031] text-primary-200">
                          <Bell className="h-5 w-5" />
                        </div>
                        <div>
                          <h2 className="text-lg font-semibold text-white">Notification Settings</h2>
                          <p className="text-sm text-white/50">Manage how you receive notifications.</p>
                        </div>
                      </div>
                    </header>

                    <div className="mt-6 space-y-4">
                      {NOTIFICATION_OPTIONS.map(({ key, label, description }) => {
                        const enabled = notificationSettings[key];
                        return (
                          <div
                            key={key}
                            className="flex items-center justify-between rounded-2xl border border-white/5 bg-[#151A27] px-5 py-4"
                          >
                            <div>
                              <p className="text-sm font-medium text-white/90">{label}</p>
                              <p className="text-xs text-white/40">{description}</p>
                            </div>
                            <button
                              type="button"
                              onClick={() => toggleNotificationSetting(key)}
                              className={`relative flex h-7 w-12 items-center rounded-full transition ${
                                enabled ? 'bg-primary-500/80' : 'bg-white/10'
                              }`}
                              aria-pressed={enabled}
                              aria-label={`${enabled ? 'Disable' : 'Enable'} ${label}`}
                            >
                              <span
                                className={`absolute h-5 w-5 rounded-full bg-white transition-transform ${
                                  enabled ? 'translate-x-[22px]' : 'translate-x-1'
                                }`}
                              />
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  </section>
                </div>
              </section>
            </div>
          ) : activeNav === 'Profile' ? (
            <div className="flex flex-1 flex-col bg-[#0F111B]">
              <header className="flex items-center justify-between border-b border-white/5 bg-[#10121C] px-6 py-6">
                <div>
                  <p className="text-xs uppercase tracking-[0.4em] text-white/40">profile</p>
                  <h1 className="mt-1 text-2xl font-semibold text-white">
                    {getProfileValue('classStream', 'Student')} {getProfileValue('division', '') ? `• Division ${getProfileValue('division', '')}` : ''}
                  </h1>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    className="rounded-full bg-[#1A1D29] p-2 text-white/60 transition hover:text-white"
                  >
                    <Moon className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    className="rounded-full bg-[#1A1D29] p-2 text-white/60 transition hover:text-white"
                  >
                    <Bell className="h-4 w-4" />
                  </button>
                  <div className="h-9 w-9 rounded-full bg-gradient-to-br from-primary-400 to-primary-600" />
                </div>
              </header>

              <section className="flex-1 overflow-y-auto px-6 py-10">
                <div className="mx-auto max-w-5xl">
                  <div className="rounded-[34px] border border-primary-500/40 bg-gradient-to-b from-primary-500/10 via-transparent to-transparent p-[1px]">
                    <div className="rounded-[32px] bg-[#0E1119] px-6 py-8 sm:px-10 sm:py-12">
                      <div className="flex flex-col items-center text-center">
                        <div className="relative h-28 w-28">
                          <div className="absolute inset-0 rounded-full bg-primary-500/40 blur-xl" />
                          <div className="relative h-full w-full overflow-hidden rounded-full border-4 border-primary-500/40 bg-gradient-to-br from-primary-400/40 via-primary-500/30 to-primary-600/40">
                            {hasPhoto ? (
                              <img
                                src={resolvedPhoto}
                                alt={resolvedFullName}
                                className="h-full w-full object-cover"
                              />
                            ) : (
                              <div className="flex h-full w-full items-center justify-center bg-[#111529] text-3xl font-semibold text-primary-100">
                                {displayInitials}
                              </div>
                            )}
                          </div>
                        </div>
                        <h2 className="mt-6 text-2xl font-semibold text-white">{resolvedFullName}</h2>
                        <p className="mt-1 text-sm text-white/60">{getProfileValue('classStream')} {getProfileValue('division') ? `• Division ${getProfileValue('division')}` : ''}</p>
                      </div>

                      <div className="mt-10 space-y-10">
                        {profileSections.map(({ title, items }) => (
                          <section key={title} className="space-y-4">
                            <div className="flex items-center gap-3 text-white/90">
                              <span className="h-1.5 w-1.5 rounded-full bg-primary-400" />
                              <h3 className="text-sm font-semibold uppercase tracking-[0.4em] text-white/60">
                                {title}
                              </h3>
                            </div>
                            <div className="grid gap-4 md:grid-cols-2">
                              {items.map(({ label, value, icon: Icon }) => (
                                <div
                                  key={label}
                                  className="flex items-start gap-3 rounded-2xl border border-white/5 bg-[#121523] px-5 py-4 shadow-lg"
                                >
                                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#1B2031] text-primary-200">
                                    <Icon className="h-4 w-4" />
                                  </div>
                                  <div>
                                    <p className="text-[11px] uppercase tracking-[0.3em] text-white/40">{label}</p>
                                    <p className="mt-2 text-sm font-medium text-white/90">{value}</p>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </section>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          ) : activeNav === 'Book' ? (
            <div className="flex flex-1 flex-col">
              <header className="flex flex-col gap-6 border-b border-white/5 bg-[#10121C] px-6 py-6 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.4em] text-white/40">{resolvedClassStream}</p>
                  <h1 className="mt-1 text-2xl font-semibold">Books</h1>
                </div>

                <div className="flex w-full flex-col gap-4 sm:flex-row sm:items-center sm:justify-end">
                  <div className="relative flex-1 max-w-xl">
                    <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-white/40" />
                    <input
                      type="search"
                      value={bookSearchTerm}
                      onChange={(event) => setBookSearchTerm(event.target.value)}
                      placeholder="Search chapters..."
                      className="w-full rounded-full bg-[#1A1D29] py-2 pl-10 pr-4 text-sm text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-primary-400/40"
                    />
                  </div>

                  <div className="flex items-center gap-3">
                    <button
                      type="button"
                      className="rounded-full bg-[#1A1D29] p-2 text-white/60 transition hover:text-white"
                    >
                      <Moon className="h-4 w-4" />
                    </button>
                    <button
                      type="button"
                      className="rounded-full bg-[#1A1D29] p-2 text-white/60 transition hover:text-white"
                    >
                      <Bell className="h-4 w-4" />
                    </button>
                    <div className="h-9 w-9 rounded-full bg-gradient-to-br from-primary-400 to-primary-600" />
                  </div>
                </div>
              </header>
              <section className="flex-1 overflow-y-auto px-6 py-8">
                <div className="flex flex-col gap-10">
                  <div className="space-y-6">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <h2 className="text-2xl font-semibold">Select Subject</h2>
                        <p className="text-sm text-white/50">Books tailored for your standard</p>
                      </div>
                      <div className="flex flex-wrap gap-2 rounded-full bg-[#141722] p-1">
                        {subjectFilters.map((subject) => (
                          <button
                            key={subject}
                            type="button"
                            onClick={() => setSelectedSubject(subject)}
                            className={`rounded-full px-4 py-1.5 text-xs transition ${
                              selectedSubject === subject
                                ? 'bg-white text-black'
                                : 'text-white/70 hover:bg-white/10'
                            }`}
                          >
                            {subject}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-lg font-semibold">My Chapters</h3>
                          <p className="text-xs text-white/40">Class {resolvedClassStream?.replace(/[^0-9A-Za-z ]/g, '') || '—'} • Student materials</p>
                        </div>
                        <span className="text-xs text-white/40">{booksLoading ? 'Loading…' : `${displayBooks.length} available`}</span>
                      </div>

                      {booksLoading ? (
                        <div className="flex items-center justify-center rounded-2xl border border-white/5 bg-[#141722] px-6 py-12 text-white/70">
                          <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Loading books…
                        </div>
                      ) : displayBooks.length ? (
                        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
                          {displayBooks.map((book) => (
                            <article
                              key={book.id}
                              className="overflow-hidden rounded-2xl border border-white/5 bg-[#141722] shadow-lg transition hover:border-primary-500/40"
                            >
                              <div className="relative">
                                <div className="flex h-48 w-full items-center justify-center bg-gradient-to-br from-[#1c1f2e] to-[#24283c] text-center text-white/70">
                                  <div>
                                    <p className="text-xs uppercase tracking-[0.4em] text-white/40">{book.subject || 'Subject'}</p>
                                    <p className="text-lg font-semibold text-white mt-2">{book.title}</p>
                                  </div>
                                </div>
                                <span className="absolute right-4 top-4 rounded-full bg-white/20 px-3 py-1 text-[11px] uppercase tracking-[0.2em]">
                                  {book.subject || 'General'}
                                </span>
                              </div>
                              <div className="space-y-2 px-5 py-5 text-sm text-white/80">
                                <div className="flex items-center justify-between text-xs text-white/50">
                                  <span>Board: {book.board || '—'}</span>
                                  <span>Chapter: {book.chapter_number || '—'}</span>
                                </div>
                                <div className="flex items-center justify-between text-xs text-white/50">
                                  <span>STD: {book.std || resolvedClassStream || '—'}</span>
                                  <span>
                                    {book.uploaded_at
                                      ? new Date(book.uploaded_at).toLocaleDateString()
                                      : ''}
                                  </span>
                                </div>
                                {book.file_url ? (
                                  <a
                                    href={book.file_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="mt-3 inline-flex w-full items-center justify-center rounded-full bg-white py-2 text-xs font-semibold text-black transition hover:scale-[1.01]"
                                  >
                                    View PDF
                                  </a>
                                ) : (
                                  <p className="mt-3 text-center text-xs text-white/40">File not available</p>
                                )}
                              </div>
                            </article>
                          ))}
                        </div>
                      ) : (
                        <div className="rounded-2xl border border-dashed border-white/10 bg-[#141722] px-6 py-16 text-center text-sm text-white/50">
                          {booksError || 'No books found for this subject.'}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </section>
            </div>
          ) : (
            <div className="flex flex-1 flex-col">
              <header className="flex flex-col gap-6 border-b border-white/5 bg-[#10121C] px-6 py-6 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.4em] text-white/40">{resolvedClassStream}</p>
                  <h1 className="mt-1 text-2xl font-semibold">Home</h1>
                  <p className="mt-1 text-sm text-white/50">Curated lectures, watch statistics, and community discussions to keep you ahead.</p>
                </div>

                <div className="flex w-full flex-col gap-4 lg:w-auto lg:flex-row lg:items-center lg:justify-end">
                  <div className="relative flex-1 max-w-xl">
                    <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-white/40" />
                    <input
                      type="search"
                      placeholder="Search lectures, chapters, or topics"
                      className="w-full rounded-full bg-[#1A1D29] py-2 pl-10 pr-4 text-sm text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-primary-400/40"
                    />
                  </div>

                  <div className="flex items-center gap-3">
                    <button type="button" className="rounded-full bg-[#1A1D29] p-2 text-white/60 transition hover:text-white">
                      <Moon className="h-4 w-4" />
                    </button>
                    <button type="button" className="rounded-full bg-[#1A1D29] p-2 text-white/60 transition hover:text-white">
                      <Bell className="h-4 w-4" />
                    </button>
                    <div className="h-9 w-9 rounded-full bg-gradient-to-br from-primary-400 to-primary-600" />
                  </div>
                </div>
              </header>
              <section className="flex-1 overflow-y-auto px-6 py-8">
                <div className="flex flex-col gap-10">
                  <div className="space-y-8">
                    <div className="overflow-hidden rounded-[30px] border border-white/5 bg-[#141722] shadow-2xl">
                      <div className="relative aspect-video w-full bg-black/40">
                        {videoLoading && !videoDetail && (
                          <div className="absolute inset-0 flex items-center justify-center text-white/70">
                            <Loader2 className="mr-2 h-6 w-6 animate-spin" /> Loading lecture…
                          </div>
                        )}

                        {videoDetail && !videoLoading && (
                          <>
                            {videoDetail.video_url ? (
                              isYouTubeUrl(videoDetail.video_url) ? (
                                <iframe
                                  key={videoDetail.id}
                                  src={getYouTubeEmbedUrl(videoDetail.video_url) ?? videoDetail.video_url}
                                  title={videoDetail.title || 'Lecture video'}
                                  className="h-full w-full"
                                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                  allowFullScreen
                                />
                              ) : (
                                <video
                                  key={videoDetail.id}
                                  controls
                                  poster={videoDetail.thumbnail_url || fallbackProfile.photoDataUrl}
                                  className="h-full w-full object-cover"
                                  preload="metadata"
                                >
                                  <source src={videoDetail.video_url} type="video/mp4" />
                                  Your browser does not support the video tag.
                                </video>
                              )
                            ) : (
                              <img
                                src={videoDetail.thumbnail_url || fallbackProfile.photoDataUrl}
                                alt={videoDetail.title}
                                className="h-full w-full object-cover"
                              />
                            )}

                            <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent" />
                            <div className="absolute bottom-6 left-6 right-6 flex flex-wrap items-center justify-between gap-4">
                              <div className="max-w-xl space-y-2">
                                <p className="text-xs uppercase tracking-[0.4em] text-white/50">{videoDetail.subject || 'Lecture'}</p>
                                <h2 className="text-3xl font-semibold leading-tight text-white">{videoDetail.title || 'Untitled Lecture'}</h2>
                                <p className="text-sm text-white/70">{videoDetail.description || 'Dive deeper into this topic and master the core concepts step-by-step.'}</p>
                              </div>
                              <div className="flex flex-wrap items-center gap-3">
                                <button
                                  type="button"
                                  onClick={toggleVideoLike}
                                  className={`flex items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] transition ${
                                    videoDetail.user_liked
                                      ? 'bg-red-500/20 text-red-200 hover:bg-red-500/30'
                                      : 'bg-white/10 text-white hover:bg-white/20'
                                  }`}
                                >
                                  {videoDetail.user_liked ? <Heart className="h-4 w-4 fill-red-400 text-red-400" /> : <HeartOff className="h-4 w-4" />}
                                  Like · {(videoDetail.total_likes ?? 0).toLocaleString('en-IN')}
                                </button>
                                <span className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-white">
                                  <Users className="h-4 w-4" />
                                  {(videoDetail.total_watch_count ?? 0).toLocaleString('en-IN')} views
                                </span>
                                <button
                                  type="button"
                                  onClick={handleOpenVideoComments}
                                  className="flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-white transition hover:bg-white/20"
                                >
                                  <MessageSquare className="h-4 w-4" /> Comments · {(videoDetail.total_comments ?? 0).toLocaleString('en-IN')}
                                </button>
                                <button
                                  type="button"
                                  onClick={handleOpenShareDialog}
                                  disabled={isSharingVideo}
                                  className="flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-white transition hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-60"
                                >
                                  <Share2 className="h-4 w-4" /> Share to Chat
                                </button>
                                <button
                                  type="button"
                                  onClick={toggleVideoSubscription}
                                  aria-pressed={videoDetail.user_subscribed}
                                  className={`flex items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] transition ${
                                    videoDetail.user_subscribed
                                      ? 'bg-primary-500/20 text-primary-100 hover:bg-primary-500/30'
                                      : 'bg-white/10 text-white hover:bg-white/20'
                                  }`}
                                >
                                  <Bookmark
                                    className={`h-4 w-4 ${videoDetail.user_subscribed ? 'fill-primary-200 text-primary-200' : ''}`}
                                  />
                                  {videoDetail.user_subscribed ? 'Saved' : 'Save'} · {savedVideoCount.toLocaleString('en-IN')}
                                </button>
                              </div>
                            </div>
                          </>
                        )}

                        {!videoDetail && !videoLoading && !videoError && (
                          <div className="absolute inset-0 flex items-center justify-center text-sm text-white/60">
                            Select a lecture to begin watching
                          </div>
                        )}

                        {videoError && !videoDetail && (
                          <div className="absolute inset-0 flex flex-col items-center justify-center px-6 text-center text-sm text-red-400">
                            <p>{videoError}</p>
                            <button
                              type="button"
                              onClick={fetchDashboardVideos}
                              className="mt-3 rounded-full bg-white/10 px-4 py-2 text-xs uppercase tracking-[0.2em] text-white transition hover:bg-white/20"
                            >
                              Retry
                            </button>
                          </div>
                        )}
                      </div>

                      {videoLoading && !videos.length ? (
                        <div className="flex items-center justify-center rounded-3xl border border-white/5 bg-[#141722] px-6 py-12 text-white/70">
                          <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Loading lectures…
                        </div>
                      ) : videos.length ? (
                        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
                          {videos.map(renderVideoCard)}
                        </div>
                      ) : (
                        <div className="rounded-3xl border border-dashed border-white/10 bg-[#141722] px-6 py-16 text-center text-sm text-white/50">
                          {videoError || 'No lectures available yet. Check back soon!'}
                        </div>
                      )}
                    </div>
                  </div>

                  <aside className="space-y-8">
                    <section className="rounded-[28px] border border-white/5 bg-[#141722] p-6 shadow-lg">
                      <header className="flex items-center justify-between">
                        <div>
                          <p className="text-xs uppercase tracking-[0.4em] text-white/40">Watch next</p>
                          <h3 className="text-base font-semibold text-white">Related lectures</h3>
                        </div>
                        <ChevronDown className="h-4 w-4 text-white/40" />
                      </header>
                      <div className="mt-5 space-y-4">
                        {relatedVideos.length ? (
                          relatedVideos.map((video) => (
                            <button
                              type="button"
                              key={video.id}
                              onClick={() => setSelectedVideoId(video.id)}
                              className={`flex w-full items-center gap-4 rounded-2xl p-3 text-left transition ${
                                selectedVideoId === video.id ? 'bg-primary-500/20 text-primary-50' : 'bg-[#10131F] hover:bg-[#181d2c]'
                              }`}
                            >
                              <img
                                src={video.thumbnail_url || fallbackProfile.photoDataUrl}
                                alt={video.title}
                                className="h-14 w-14 rounded-xl object-cover"
                              />
                              <div className="flex-1">
                                <p className="text-xs uppercase tracking-[0.3em] text-white/40">{video.subject || 'Lecture'}</p>
                                <p className="mt-1 text-sm font-semibold leading-tight text-white line-clamp-2">{video.title || 'Untitled Lecture'}</p>
                                <p className="mt-1 text-xs text-white/40">
                                  {(video.total_watch_count ?? 0).toLocaleString('en-IN')} views ·
                                  {' '}{video.duration_seconds ? `${Math.floor(video.duration_seconds / 60)}:${String(video.duration_seconds % 60).padStart(2, '0')}` : '—'}
                                </p>
                              </div>
                            </button>
                          ))
                        ) : (
                          <p className="rounded-2xl border border-dashed border-white/10 bg-[#10131F] p-4 text-center text-xs text-white/40">
                            {videos.length ? 'Select another lecture to see more suggestions.' : 'Upload or view lectures to build recommendations.'}
                          </p>
                        )}
                      </div>
                    </section>

                    <section className="rounded-[28px] border border-white/5 bg-[#141722] p-6 shadow-lg">
                      <header className="flex items-center justify-between">
                        <div>
                          <p className="text-xs uppercase tracking-[0.4em] text-white/40">Progress</p>
                          <h3 className="text-base font-semibold text-white">Watched lectures</h3>
                        </div>
                        <button type="button" onClick={() => setActiveNav('Watched Lecture')} className="text-xs uppercase tracking-[0.3em] text-white/40 transition hover:text-white">
                          View all
                        </button>
                      </header>
                      <div className="mt-5 space-y-4">
                        <div className="rounded-2xl bg-black/20 p-4">
                          <div className="flex items-center justify-between text-xs text-white/50">
                            <span>Completed</span>
                            <span>{completionPercent}%</span>
                          </div>
                          <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-white/10">
                            <div className="h-full rounded-full bg-primary-500" style={{ width: `${completionPercent}%` }} />
                          </div>
                          <p className="mt-3 text-xs text-white/50">
                            {(watchedSummary.completed_videos ?? 0).toLocaleString('en-IN')} of {(watchedSummary.total_records ?? 0).toLocaleString('en-IN')} lectures completed
                          </p>
                          <p className="text-xs text-white/50">Total watch time · {formatWatchTime}</p>
                        </div>
                        {watchedLoading ? (
                          <div className="flex items-center justify-center rounded-2xl border border-white/5 bg-[#10131F] px-4 py-6 text-white/70">
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading history…
                          </div>
                        ) : watchedVideos.length ? (
                          watchedVideos.slice(0, 3).map((video) => (
                            <div key={video.id} className="flex items-center gap-3 rounded-2xl border border-white/5 bg-[#10131F] p-3">
                              <img
                                src={video.thumbnail_url || fallbackProfile.photoDataUrl}
                                alt={video.title}
                                className="h-12 w-12 rounded-xl object-cover"
                              />
                              <div className="flex-1">
                                <p className="text-sm font-semibold text-white line-clamp-1">{video.title || 'Untitled Lecture'}</p>
                                <p className="text-xs text-white/40">
                                  Watched on{' '}
                                  {video.last_watched_at
                                    ? new Date(video.last_watched_at).toLocaleDateString()
                                    : 'N/A'}
                                </p>
                              </div>
                              <span className="text-xs text-white/50">{video.duration_seconds ? `${Math.floor(video.duration_seconds / 60)}:${String(video.duration_seconds % 60).padStart(2, '0')}` : '—'}</span>
                            </div>
                          ))
                        ) : (
                          <p className="rounded-2xl border border-dashed border-white/10 bg-[#10131F] p-4 text-center text-xs text-white/40">
                            {watchedError || 'Watch lectures to see your progress summary here.'}
                          </p>
                        )}
                      </div>
                    </section>
                  </aside>
                </div>
              </section>

              {/* Share dialog */}
              <div
                className={`fixed inset-0 z-50 px-4 py-6 transition ${
                  isShareDialogOpen ? 'pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'
                }`}
              >
                <div
                  className={`absolute inset-0 bg-black/70 backdrop-blur-sm transition-opacity duration-300 ${
                    isShareDialogOpen ? 'opacity-100' : 'opacity-0'
                  }`}
                  onClick={handleCloseShareDialog}
                  aria-hidden="true"
                />
                <div
                  className={`relative z-10 mx-auto flex h-full max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-3xl border border-white/10 bg-[#11131F] shadow-2xl transition-all duration-300 ${
                    isShareDialogOpen ? 'translate-y-0 opacity-100' : 'translate-y-6 opacity-0'
                  }`}
                  role="dialog"
                  aria-modal="true"
                >
                  <header className="flex items-center justify-between border-b border-white/10 px-6 py-5">
                    <div>
                      <p className="text-xs uppercase tracking-[0.4em] text-white/40">Share lecture</p>
                      <h3 className="mt-1 text-lg font-semibold text-white">Send to classmates</h3>
                    </div>
                    <button
                      type="button"
                      onClick={handleCloseShareDialog}
                      className="rounded-full bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.3em] text-white transition hover:bg-white/20"
                    >
                      Close
                    </button>
                  </header>
                  <div className="border-b border-white/10 px-6 py-5">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-white/40" />
                      <input
                        type="search"
                        value={shareSearchTerm}
                        onChange={(event) => setShareSearchTerm(event.target.value)}
                        placeholder="Search student or enrollment..."
                        className="w-full rounded-full bg-[#1A1D29] py-2 pl-10 pr-4 text-sm text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-primary-400/40"
                      />
                    </div>
                    <p className="mt-4 text-xs text-white/40">
                      Select one or more students from your class to share this lecture.
                    </p>
                  </div>
                  <div className="flex-1 overflow-y-auto px-6 py-5">
                    {peersLoading && !peers.length ? (
                      <div className="flex h-full min-h-[30vh] items-center justify-center text-white/70">
                        <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Loading classmates…
                      </div>
                    ) : shareFilteredPeers.length ? (
                      <div className="space-y-3">
                        {shareFilteredPeers.map((peer) => {
                          const isSelected = shareSelectedRecipients.includes(peer.enrollment_number);
                          return (
                            <button
                              key={peer.enrollment_number}
                              type="button"
                              onClick={() => handleToggleShareRecipient(peer.enrollment_number)}
                              className={`flex w-full items-center gap-4 rounded-2xl border px-4 py-3 text-left transition ${
                                isSelected
                                  ? 'border-primary-500/40 bg-primary-500/10 text-primary-100'
                                  : 'border-white/5 bg-[#161926] text-white/80 hover:border-white/10 hover:bg-[#1C2030]'
                              }`}
                            >
                              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-primary-400 to-primary-600 text-sm font-semibold text-white">
                                {getInitials(peer.name)}
                              </div>
                              <div className="flex-1">
                                <p className="text-sm font-semibold text-white/90">{peer.name}</p>
                                <p className="text-xs text-white/40">
                                  {peer.enrollment_number || 'Enrollment unavailable'}
                                  {peer.division ? ` · Division ${peer.division}` : ''}
                                </p>
                              </div>
                              <div
                                className={`flex h-6 w-6 items-center justify-center rounded-full border ${
                                  isSelected ? 'border-primary-400 bg-primary-500/70 text-black' : 'border-white/20 text-transparent'
                                }`}
                              >
                                {isSelected ? <Check className="h-4 w-4" /> : null}
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="rounded-2xl border border-dashed border-white/10 bg-[#161926] px-5 py-10 text-center text-sm text-white/50">
                        {chatError || 'No classmates found to share with yet.'}
                      </div>
                    )}
                  </div>
                  <div className="border-t border-white/10 bg-[#10131F] px-6 py-5">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <p className="text-xs text-white/50">
                        {shareSelectedRecipients.length
                          ? `${shareSelectedRecipients.length} student${shareSelectedRecipients.length > 1 ? 's' : ''} selected`
                          : 'No students selected yet.'}
                      </p>
                      <div className="flex flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={handleCloseShareDialog}
                          className="rounded-full bg-white/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-white transition hover:bg-white/20"
                        >
                          Cancel
                        </button>
                        <button
                          type="button"
                          onClick={shareVideoToSelectedPeers}
                          disabled={isSharingVideo || !shareSelectedRecipients.length}
                          className="inline-flex items-center gap-2 rounded-full bg-primary-500 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-black transition hover:bg-primary-400 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          <Share2 className="h-4 w-4" /> {isSharingVideo ? 'Sharing…' : 'Share'}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Comments drawer */}
              <div
                className={`pointer-events-none fixed inset-y-0 right-0 z-40 flex w-full max-w-xl transform bg-[#0E1018]/80 backdrop-blur transition-transform duration-300 ${
                  commentsOpen ? 'translate-x-0' : 'translate-x-full'
                }`}
              >
                <div
                  className={`pointer-events-auto flex h-full w-full flex-col border-l border-white/10 bg-[#0F121D]`}
                >
                  <header className="flex items-center justify-between border-b border-white/10 px-6 py-5">
                    <div>
                      <p className="text-xs uppercase tracking-[0.4em] text-white/40">Comments</p>
                      <h3 className="text-lg font-semibold text-white">Join the discussion</h3>
                    </div>
                    <button type="button" onClick={() => setCommentsOpen(false)} className="rounded-full bg-white/10 px-3 py-1 text-xs uppercase tracking-[0.3em] text-white transition hover:bg-white/20">
                      Close
                    </button>
                  </header>
                  <div ref={commentListRef} className="flex-1 space-y-5 overflow-y-auto px-6 py-6">
                    {drawerComments.length ? (
                      drawerComments.map((comment) => (
                        <div key={comment.id} className="rounded-2xl border border-white/5 bg-[#13172A] px-5 py-4">
                          <div className="flex items-center justify-between text-xs text-white/40">
                            <span>{comment.student_name || comment.enrollment_number || 'Anonymous'}</span>
                            <time dateTime={comment.created_at}>
                              {comment.created_at ? new Date(comment.created_at).toLocaleString() : ''}
                            </time>
                          </div>
                          <p className="mt-3 text-sm text-white/80">{comment.comment}</p>
                        </div>
                      ))
                    ) : (
                      <p className="text-center text-sm text-white/40">No comments yet. Be the first to share a thought!</p>
                    )}
                  </div>
                  <form onSubmit={handleSubmitComment} className="border-t border-white/10 bg-[#0F121D] px-6 py-5">
                    <div className="flex items-center gap-3 rounded-2xl bg-[#1A1D29] px-4 py-3">
                      <input
                        type="text"
                        value={commentDraft}
                        onChange={(event) => setCommentDraft(event.target.value)}
                        placeholder="Share your thoughts"
                        className="flex-1 bg-transparent text-sm text-white placeholder:text-white/40 focus:outline-none"
                      />
                      <button
                        type="submit"
                        disabled={isSubmittingComment}
                        className="inline-flex items-center gap-2 rounded-full bg-primary-500 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-black transition hover:bg-primary-400 disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {isSubmittingComment ? 'Posting…' : 'Post'}
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default StudentPortalDashboard;