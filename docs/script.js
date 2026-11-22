(() => {
  const copyButtons = document.querySelectorAll('.copy-btn');

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
    const helper = document.createElement('textarea');
    helper.value = text;
    helper.setAttribute('readonly', '');
    helper.style.position = 'fixed';
    helper.style.opacity = '0';
    document.body.appendChild(helper);
    helper.select();
    let ok = false;
    try {
      ok = document.execCommand('copy');
    } finally {
      document.body.removeChild(helper);
    }
    return ok;
  }

  copyButtons.forEach((button) => {
    const defaultLabel = button.textContent.trim();
    let timer = null;

    button.addEventListener('click', async () => {
      const targetId = button.getAttribute('data-copy-target');
      const targetEl = targetId ? document.getElementById(targetId) : null;
      if (!targetEl) return;

      const text = targetEl.textContent.trim();
      try {
        const copied = await copyText(text);
        if (!copied) return;
        button.textContent = 'Copied';
        button.setAttribute('data-copied', 'true');
        if (timer) window.clearTimeout(timer);
        timer = window.setTimeout(() => {
          button.textContent = defaultLabel;
          button.removeAttribute('data-copied');
        }, 2000);
      } catch (_) {
        // Ignore clipboard errors gracefully.
      }
    });
  });

  // Native <dialog> lightbox for inspecting high-resolution figures
  const dialog = document.getElementById('figure-lightbox');
  const dialogImg = document.getElementById('lightbox-img');
  const dialogTitle = document.getElementById('lightbox-title');
  const dialogCaption = document.getElementById('lightbox-caption');
  const closeBtn = document.getElementById('lightbox-close');

  if (dialog && typeof dialog.showModal === 'function' && dialogImg && dialogTitle && dialogCaption) {
    document.querySelectorAll('.figure-zoom').forEach((trigger) => {
      trigger.addEventListener('click', () => {
        const img = trigger.querySelector('img');
        const figure = trigger.closest('figure');
        const caption = figure ? figure.querySelector('figcaption') : null;
        if (!img) return;

        dialogImg.src = img.getAttribute('src') || '';
        dialogImg.alt = img.getAttribute('alt') || '';
        dialogTitle.textContent = trigger.getAttribute('data-title') || img.getAttribute('alt') || 'Figure';
        dialogCaption.textContent = caption ? caption.textContent.trim() : '';
        dialog.showModal();
      });
    });

    if (closeBtn) {
      closeBtn.addEventListener('click', () => dialog.close());
    }

    dialog.addEventListener('click', (event) => {
      if (event.target === dialog) {
        dialog.close();
      }
    });

    dialog.addEventListener('close', () => {
      dialogImg.removeAttribute('src');
    });
  }

  // Subtle active navigation indicator
  const navLinks = Array.from(document.querySelectorAll('.site-nav a[href^="#"]'));
  const topSection = document.getElementById('top');
  const observedSections = navLinks
    .map((link) => {
      const id = link.getAttribute('href').slice(1);
      const section = document.getElementById(id);
      return section ? { link, section } : null;
    })
    .filter(Boolean);

  if ('IntersectionObserver' in window && observedSections.length > 0) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          if (entry.target === topSection) {
            observedSections.forEach(({ link }) => link.removeAttribute('aria-current'));
            return;
          }
          observedSections.forEach(({ link, section }) => {
            if (section === entry.target) {
              link.setAttribute('aria-current', 'location');
            } else {
              link.removeAttribute('aria-current');
            }
          });
        });
      },
      { rootMargin: '-25% 0px -60% 0px' }
    );

    if (topSection) observer.observe(topSection);
    observedSections.forEach(({ section }) => observer.observe(section));
  }

  // Video chapter seek & active-chapter sync
  const video = document.getElementById('save-overview-video');
  const chapterButtons = Array.from(
    document.querySelectorAll('.video-chapter-btn[data-seek]')
  );

  if (video && chapterButtons.length > 0) {
    const chapters = chapterButtons
      .map((btn) => ({
        btn,
        seek: Number.parseFloat(btn.getAttribute('data-seek') || '0'),
      }))
      .sort((a, b) => a.seek - b.seek);

    const setActiveChapter = (activeBtn) => {
      chapterButtons.forEach((btn) => {
        btn.setAttribute('aria-pressed', btn === activeBtn ? 'true' : 'false');
      });
    };

    const seekTo = (seconds) => {
      const applySeek = () => {
        video.currentTime = seconds;
        const playPromise = video.play();
        if (playPromise && typeof playPromise.catch === 'function') {
          playPromise.catch(() => {});
        }
      };

      if (video.readyState >= 1) {
        applySeek();
      } else {
        video.addEventListener('loadedmetadata', applySeek, { once: true });
        video.load();
      }
    };

    chapters.forEach(({ btn, seek }) => {
      btn.addEventListener('click', () => {
        setActiveChapter(btn);
        seekTo(seek);
      });
    });

    video.addEventListener('timeupdate', () => {
      const now = video.currentTime;
      let current = chapters[0];
      for (const ch of chapters) {
        if (now + 0.15 >= ch.seek) {
          current = ch;
        }
      }
      if (current) {
        setActiveChapter(current.btn);
      }
    });
  }
})();
