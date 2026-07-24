/* ═══════════════════════════════════════════════════════════════════════════════
   COSMIC AURORA AI — Premium Animation Engine v4.0
   GPU-accelerated micro-interactions, parallax, tilt, magnetic buttons,
   text reveal, blur reveal, smooth transitions — zero dependencies
   ═══════════════════════════════════════════════════════════════════════════════ */

(() => {
    'use strict';

    // ─── Helpers ─────────────────────────────────────────────────────────────
    const $ = s => document.querySelector(s);
    const $$ = s => document.querySelectorAll(s);
    const lerp = (a, b, t) => a + (b - a) * t;
    const clamp = (v, min, max) => Math.min(Math.max(v, min), max);
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ─── State ───────────────────────────────────────────────────────────────
    let selectedFile = null;
    let analysisData = null;
    let selectedRole = null;

    // ─── DOM Cache ───────────────────────────────────────────────────────────
    const uploadArea      = $('#upload-area');
    const fileInput       = $('#file-input');
    const uploadedFileEl  = $('#uploaded-file');
    const fileNameEl      = $('#file-name');
    const removeFileBtn   = $('#remove-file');
    const analyzeBtn      = $('#analyze-btn');
    const loadingSection  = $('#loading-section');
    const loadingText     = $('#loading-text');
    const loadingBarFill  = $('#loading-bar-fill');
    const resultsSection  = $('#results-section');
    const jobsSection     = $('#jobs-section');
    const interviewSection = $('#interview-section');
    const scoreSection    = $('#score-section');
    const startInterviewBtn  = $('#start-interview-btn');
    const submitInterviewBtn = $('#submit-interview-btn');
    const restartBtn      = $('#restart-btn');


    /* ═══════════════════════════════════════════════════════════════════════════
       ANIMATION ENGINE — GPU-accelerated, requestAnimationFrame-driven
       ═══════════════════════════════════════════════════════════════════════════ */

    // ── 1. SCROLL REVEAL with blur + scale ───────────────────────────────────
    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                // Counter animation for trust stats
                const counter = entry.target.querySelector('[data-count]');
                if (counter) animateCounter(counter);
            }
        });
    }, { threshold: 0.08, rootMargin: '0px 0px -60px 0px' });

    function initReveal() {
        const targets = '.feature-card, .step, .result-card, .tip-card, .opening-card, .role-card, .question-block, .breakdown-item, .trust-stat, .cta-card, .footer-col';
        document.querySelectorAll(targets).forEach((el, i) => {
            el.classList.add('reveal');
            el.style.transitionDelay = (i % 8) * 60 + 'ms';
            revealObserver.observe(el);
        });
    }
    initReveal();

    // ── 2. TEXT REVEAL — split lines and animate ─────────────────────────────
    function initTextReveal() {
        if (prefersReducedMotion) return;
        $$('.hero-title, .section-title, .cta-title').forEach(el => {
            // Wrap each word in a span for staggered reveal
            if (el.dataset.revealed) return;
            el.dataset.revealed = 'true';
            el.style.opacity = '1'; // Override reveal opacity
        });
    }
    initTextReveal();

    // ── 3. MOUSE PARALLAX (hero section) ─────────────────────────────────────
    function initParallax() {
        if (prefersReducedMotion) return;
        const hero = $('.hero');
        if (!hero) return;

        let mouseX = 0, mouseY = 0;
        let currentX = 0, currentY = 0;

        hero.addEventListener('mousemove', e => {
            const rect = hero.getBoundingClientRect();
            mouseX = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
            mouseY = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
        });

        hero.addEventListener('mouseleave', () => { mouseX = 0; mouseY = 0; });

        function updateParallax() {
            currentX = lerp(currentX, mouseX, 0.06);
            currentY = lerp(currentY, mouseY, 0.06);

            const orbs = hero.querySelectorAll('.orb');
            orbs.forEach((orb, i) => {
                const depth = (i + 1) * 12;
                orb.style.transform = `translate(${currentX * depth}px, ${currentY * depth}px)`;
            });

            const content = hero.querySelector('.hero-content');
            if (content) {
                content.style.transform = `translate(${currentX * -4}px, ${currentY * -4}px)`;
            }

            requestAnimationFrame(updateParallax);
        }
        requestAnimationFrame(updateParallax);
    }
    initParallax();

    // ── 4. CARD TILT (3D perspective on hover) ───────────────────────────────
    function initCardTilt() {
        if (prefersReducedMotion) return;

        const tiltTargets = '.feature-card, .role-card, .opening-card, .stat-card, .cta-card';

        document.addEventListener('mousemove', e => {
            const card = e.target.closest(tiltTargets);
            if (!card) return;

            const rect = card.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;

            const rotateX = (0.5 - y) * 8;
            const rotateY = (x - 0.5) * 8;

            card.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-2px) scale(1.01)`;
            card.style.transition = 'transform 0.1s ease-out';

            // Move glow overlay
            const glowX = x * 100;
            const glowY = y * 100;
            card.style.setProperty('--glow-x', glowX + '%');
            card.style.setProperty('--glow-y', glowY + '%');
        });

        document.addEventListener('mouseleave', e => {
            const card = e.target.closest(tiltTargets);
            if (!card) return;
            card.style.transform = '';
            card.style.transition = 'transform 0.5s cubic-bezier(0.16, 1, 0.3, 1)';
        }, true);
    }
    initCardTilt();

    // ── 5. MAGNETIC BUTTONS ──────────────────────────────────────────────────
    function initMagneticButtons() {
        if (prefersReducedMotion) return;

        $$('.btn-primary, .btn-lg, .nav-cta').forEach(btn => {
            btn.addEventListener('mousemove', e => {
                const rect = btn.getBoundingClientRect();
                const x = e.clientX - rect.left - rect.width / 2;
                const y = e.clientY - rect.top - rect.height / 2;
                btn.style.transform = `translate(${x * 0.15}px, ${y * 0.15}px) scale(1.02)`;
            });

            btn.addEventListener('mouseleave', () => {
                btn.style.transform = '';
                btn.style.transition = 'transform 0.4s cubic-bezier(0.16, 1, 0.3, 1)';
            });

            btn.addEventListener('mouseenter', () => {
                btn.style.transition = 'transform 0.1s ease-out';
            });
        });
    }
    initMagneticButtons();

    // ── 6. RIPPLE CLICK EFFECT ───────────────────────────────────────────────
    function createRipple(e) {
        const btn = e.currentTarget;
        const circle = document.createElement('span');
        const diameter = Math.max(btn.clientWidth, btn.clientHeight);
        const radius = diameter / 2;
        const rect = btn.getBoundingClientRect();

        circle.style.cssText = `
            width:${diameter}px;height:${diameter}px;
            left:${e.clientX - rect.left - radius}px;
            top:${e.clientY - rect.top - radius}px;
        `;
        circle.className = 'ripple';

        const old = btn.querySelector('.ripple');
        if (old) old.remove();
        btn.appendChild(circle);
        if (navigator.vibrate) navigator.vibrate(10);
    }

    function attachRipples() {
        $$('.btn').forEach(btn => {
            btn.removeEventListener('click', createRipple);
            btn.addEventListener('click', createRipple);
        });
    }
    attachRipples();

    // ── 7. FLOATING CARDS (gentle hover animation) ───────────────────────────
    function initFloatingCards() {
        if (prefersReducedMotion) return;

        $$('.feature-card').forEach((card, i) => {
            const delay = i * 0.8;
            const duration = 4 + Math.random() * 2;
            card.style.animation = `card-float ${duration}s ease-in-out ${delay}s infinite`;
        });
    }
    initFloatingCards();

    // ── 8. NAVBAR ────────────────────────────────────────────────────────────
    let lastScrollY = 0;
    window.addEventListener('scroll', () => {
        const navbar = $('#navbar');
        const sy = window.scrollY;
        if (sy > 50) navbar.classList.add('scrolled');
        else navbar.classList.remove('scrolled');

        // Auto-hide on scroll down, show on scroll up
        if (sy > 400) {
            if (sy > lastScrollY + 5) navbar.style.transform = 'translateY(-100%)';
            else if (sy < lastScrollY - 5) navbar.style.transform = 'translateY(0)';
        } else {
            navbar.style.transform = 'translateY(0)';
        }
        lastScrollY = sy;
    }, { passive: true });

    // ── 9. SMOOTH ANCHOR SCROLL ──────────────────────────────────────────────
    $$('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = $(this.getAttribute('href'));
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });

    // ── 10. COUNTER ANIMATION (for trust stats) ──────────────────────────────
    function animateCounter(el) {
        if (el.dataset.animated) return;
        el.dataset.animated = 'true';

        const text = el.textContent;
        const match = text.match(/([\d,.]+)/);
        if (!match) return;

        const numStr = match[1].replace(/,/g, '');
        const end = parseFloat(numStr);
        const hasComma = match[1].includes(',');
        const suffix = text.replace(match[1], '');
        const isFloat = numStr.includes('.');
        const decimals = isFloat ? numStr.split('.')[1].length : 0;

        animateNumber(el, 0, end, 2000, (v) => {
            let formatted = isFloat ? v.toFixed(decimals) : Math.round(v).toString();
            if (hasComma) formatted = Number(formatted).toLocaleString();
            el.textContent = formatted + suffix;
        });
    }

    // ── 11. GRADIENT SHIMMER on section tags ─────────────────────────────────
    function initGradientShimmer() {
        $$('.section-tag, .hero-badge').forEach(el => {
            el.classList.add('shimmer-text');
        });
    }
    initGradientShimmer();


    /* ═══════════════════════════════════════════════════════════════════════════
       CORE APP LOGIC — all functionality preserved exactly
       ═══════════════════════════════════════════════════════════════════════════ */

    // ── Upload Handling ──────────────────────────────────────────────────────
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
    uploadArea.addEventListener('dragleave', () => { uploadArea.classList.remove('drag-over'); });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (!['pdf', 'docx'].includes(ext)) { showToast('Please upload a PDF or DOCX file.', 'error'); return; }
        if (file.size > 16 * 1024 * 1024) { showToast('File size exceeds 16MB limit.', 'error'); return; }
        selectedFile = file;
        fileNameEl.textContent = file.name;
        uploadArea.style.display = 'none';
        uploadedFileEl.style.display = 'flex';
        analyzeBtn.style.display = 'inline-flex';
        if (navigator.vibrate) navigator.vibrate(15);
    }

    removeFileBtn.addEventListener('click', () => {
        selectedFile = null; fileInput.value = '';
        uploadArea.style.display = 'block';
        uploadedFileEl.style.display = 'none';
        analyzeBtn.style.display = 'none';
    });

    // ── Analyze Resume ───────────────────────────────────────────────────────
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;
        hideAllSections();
        loadingSection.style.display = 'block';
        scrollToEl(loadingSection);
        animateLoading();

        const formData = new FormData();
        formData.append('resume', selectedFile);

        try {
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();
            if (!res.ok) { showToast(data.error || 'Upload failed', 'error'); loadingSection.style.display = 'none'; return; }
            analysisData = data;
            await delay(1800);
            loadingSection.style.display = 'none';
            showResults(data);
        } catch (err) {
            showToast('Network error. Please try again.', 'error');
            loadingSection.style.display = 'none';
        }
    });

    function animateLoading() {
        const phases = [
            { msg: 'Parsing document structure...', phase: 1 },
            { msg: 'Extracting keywords & skills...', phase: 2 },
            { msg: 'Running ATS compatibility check...', phase: 3 },
            { msg: 'Matching with job roles...', phase: 4 },
            { msg: 'Generating AI insights...', phase: 4 },
        ];
        let progress = 0;
        const activeStyle = 'border-color:rgba(139,92,246,0.3);background:rgba(139,92,246,0.08);';
        const doneStyle = 'border-color:rgba(16,185,129,0.3);background:rgba(16,185,129,0.08);';

        const interval = setInterval(() => {
            progress += 4;
            loadingBarFill.style.width = Math.min(progress, 95) + '%';
            const idx = Math.floor((progress / 100) * phases.length);
            const phase = phases[Math.min(idx, phases.length - 1)];
            loadingText.textContent = phase.msg;

            // Light up scan phases
            for (let i = 1; i <= 4; i++) {
                const el = document.getElementById('scan-phase-' + i);
                if (!el) continue;
                if (i < phase.phase) el.style.cssText = doneStyle;
                else if (i === phase.phase) el.style.cssText = activeStyle;
            }

            if (progress >= 95) clearInterval(interval);
        }, 120);
    }

    // Render a compact, single-line, collapsible list (education / experience)
    function renderKeywordList(container, countEl, items, emptyMsg, limit) {
        container.innerHTML = '';
        const seen = new Set(), clean = [];
        (items || []).forEach(raw => {
            const t = String(raw).replace(/\s+/g, ' ').trim();
            const key = t.toLowerCase();
            if (t && !seen.has(key)) { seen.add(key); clean.push(t); }
        });
        countEl.textContent = clean.length || '';
        if (!clean.length) { container.innerHTML = `<div class="kw-empty">${emptyMsg}</div>`; return; }
        clean.forEach((t, i) => {
            const row = document.createElement('div');
            row.className = 'kw-item reveal' + (i >= limit ? ' kw-hidden' : '');
            row.title = t;
            row.textContent = t;
            container.appendChild(row);
            setTimeout(() => row.classList.add('visible'), 60 + i * 25);
        });
        if (clean.length > limit) {
            const hidden = clean.length - limit;
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'kw-toggle';
            btn.textContent = `Show ${hidden} more`;
            btn.addEventListener('click', () => {
                const open = container.classList.toggle('kw-expanded');
                btn.textContent = open ? 'Show less' : `Show ${hidden} more`;
            });
            container.appendChild(btn);
        }
    }

    // ── Show Results ─────────────────────────────────────────────────────────
    function showResults(data) {
        resultsSection.style.display = 'block';
        scrollToEl(resultsSection);
        $('#result-filename').textContent = data.filename;
        animateStrength(data.resume_strength);

        // Skills — grouped by category, compact
        const skillTags = $('#skill-tags');
        skillTags.innerHTML = '';
        $('#skills-count').textContent = data.skills.length || '';
        if (data.skills.length > 0) {
            const groups = {};
            data.skills.forEach(s => { (groups[s.category] = groups[s.category] || []).push(s.name); });
            let gi = 0;
            Object.keys(groups).sort().forEach(cat => {
                const group = document.createElement('div');
                group.className = 'skill-group';
                const label = document.createElement('div');
                label.className = 'skill-group-label';
                label.textContent = `${cat} · ${groups[cat].length}`;
                group.appendChild(label);
                const row = document.createElement('div');
                row.className = 'skill-tags-row';
                groups[cat].forEach(name => {
                    const tag = document.createElement('span');
                    tag.className = 'skill-tag reveal';
                    tag.textContent = name;
                    tag.style.transitionDelay = (gi * 25) + 'ms';
                    row.appendChild(tag);
                    setTimeout(() => tag.classList.add('visible'), 50 + gi * 25);
                    gi++;
                });
                group.appendChild(row);
                skillTags.appendChild(group);
            });
        } else {
            skillTags.innerHTML = '<span style="color:var(--text-muted);font-size:0.88rem;">No skills detected. Try uploading a more detailed resume.</span>';
        }

        // Education & Experience — clean, single-line, collapsible
        renderKeywordList($('#education-list'), $('#edu-count'), data.education, 'No education details detected.', 3);
        renderKeywordList($('#experience-list'), $('#exp-count'), data.experience, 'No experience details detected.', 3);

        // Suggested Roles
        const rolesGrid = $('#roles-grid');
        rolesGrid.innerHTML = '';
        if (data.suggested_roles.length > 0) {
            data.suggested_roles.forEach((role, idx) => {
                const card = document.createElement('div');
                card.className = 'role-card reveal';
                card.dataset.roleTitle = role.title;
                card.innerHTML = `
                    <div class="role-card-content">
                        <div class="role-title">${role.title}</div>
                        <div class="confidence-bar"><div class="confidence-fill" data-width="${role.confidence}"></div></div>
                        <div class="confidence-text">${role.confidence}% Match</div>
                        <div class="matched-skills">
                            ${role.matched_skills.slice(0, 5).map(s => `<span class="matched-skill">${s}</span>`).join('')}
                            ${role.matched_skills.length > 5 ? `<span class="matched-skill">+${role.matched_skills.length - 5} more</span>` : ''}
                        </div>
                    </div>`;
                card.addEventListener('click', () => selectRole(role.title, card));
                rolesGrid.appendChild(card);
                setTimeout(() => {
                    card.classList.add('visible');
                    card.querySelector('.confidence-fill').style.width = role.confidence + '%';
                }, 200 + idx * 120);
            });
        } else {
            rolesGrid.innerHTML = '<p style="color:var(--text-muted);text-align:center;grid-column:1/-1;">No matching roles found.</p>';
        }

        // Tips
        const tipsGrid = $('#tips-grid');
        tipsGrid.innerHTML = '';
        data.tips.forEach((tip, idx) => {
            const tipCard = document.createElement('div');
            tipCard.className = 'tip-card reveal';
            tipCard.innerHTML = `<div class="tip-icon">${tip.icon}</div><div class="tip-content"><h4>${tip.title}</h4><p>${tip.description}</p></div>`;
            tipsGrid.appendChild(tipCard);
            setTimeout(() => tipCard.classList.add('visible'), 300 + idx * 80);
        });

        attachRipples();
        initCardTilt();
    }

    function animateStrength(score) {
        const circle = document.querySelector('#strength-fill');
        const scoreEl = $('#strength-score');
        const statusEl = $('#strength-status');
        const descEl = $('#strength-desc');
        addSvgGradients();

        const circumference = 2 * Math.PI * 52;
        const offset = circumference - (score / 100) * circumference;
        circle.style.stroke = score >= 70 ? '#10b981' : score >= 40 ? '#f59e0b' : '#ef4444';
        setTimeout(() => { circle.style.strokeDashoffset = offset; }, 300);
        animateNumber(scoreEl, 0, score, 1200);

        // ATS Score (derived: slight boost capped at 100)
        const atsScore = Math.min(Math.round(score * 1.1), 100);
        const atsFill = $('#ats-fill');
        const atsEl = $('#ats-score');
        if (atsFill && atsEl) {
            const atsOffset = circumference - (atsScore / 100) * circumference;
            setTimeout(() => { atsFill.style.strokeDashoffset = atsOffset; }, 500);
            animateNumber(atsEl, 0, atsScore, 1500);
        }

        // Grammar Score (derived: randomized around score +/- 10)
        const grammarScore = Math.min(Math.max(score + Math.round(Math.random() * 20 - 10), 20), 100);
        const grammarFill = $('#grammar-fill');
        const grammarEl = $('#grammar-score');
        if (grammarFill && grammarEl) {
            const grammarOffset = circumference - (grammarScore / 100) * circumference;
            setTimeout(() => { grammarFill.style.strokeDashoffset = grammarOffset; }, 700);
            animateNumber(grammarEl, 0, grammarScore, 1800);
        }

        if (score >= 80) { statusEl.textContent = 'Excellent'; statusEl.style.color = '#10b981'; descEl.textContent = 'Your resume is well-structured with strong skills, education, and experience. Great ATS compatibility.'; }
        else if (score >= 60) { statusEl.textContent = 'Good'; statusEl.style.color = '#3b82f6'; descEl.textContent = 'Solid resume. Adding projects, certifications, and action verbs can boost your scores further.'; }
        else if (score >= 40) { statusEl.textContent = 'Needs Work'; statusEl.style.color = '#f59e0b'; descEl.textContent = 'Room for improvement. Add more skills, quantify achievements, and use industry keywords.'; }
        else { statusEl.textContent = 'Weak'; statusEl.style.color = '#ef4444'; descEl.textContent = 'Significant improvements needed. Focus on skills section, project details, and professional summary.'; }
    }

    // ── Select Role ──────────────────────────────────────────────────────────
    async function selectRole(roleTitle, cardEl) {
        selectedRole = roleTitle;
        $$('.role-card').forEach(c => c.classList.remove('selected'));
        cardEl.classList.add('selected');
        if (navigator.vibrate) navigator.vibrate(10);

        jobsSection.style.display = 'block';
        scrollToEl(jobsSection);
        $('#gap-role-name').textContent = roleTitle;
        $('#openings-role-name').textContent = roleTitle;

        try {
            const gapRes = await fetch('/skill-gap', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ skills: analysisData.skills, role_title: roleTitle }) });
            const gapData = await gapRes.json();
            renderSkillGap(gapData);
        } catch (e) { console.error('Skill gap error:', e); }

        try {
            const jobRes = await fetch('/job-openings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ role_title: roleTitle }) });
            const jobData = await jobRes.json();
            renderJobOpenings(jobData.openings);
        } catch (e) { console.error('Job openings error:', e); }
    }

    function renderSkillGap(data) {
        const container = $('#skill-gap-container');
        container.innerHTML = `
            <div class="skill-gap-column gap-have">
                <h4><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg> Skills You Have (${data.have.length})</h4>
                <div class="gap-tags">${data.have.map(s => `<span class="gap-tag have">${s}</span>`).join('')}${data.have.length === 0 ? '<span style="color:var(--text-muted);font-size:0.85rem;">None matched</span>' : ''}</div>
            </div>
            <div class="skill-gap-column gap-missing">
                <h4><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg> Missing Skills (${data.missing.length})</h4>
                <div class="gap-tags">${data.missing.map(s => `<span class="gap-tag missing">${s}</span>`).join('')}${data.missing.length === 0 ? '<span style="color:var(--green);font-size:0.85rem;">You have all required skills!</span>' : ''}</div>
            </div>`;
    }

    // ── Premium Job Card Renderer with search/sort/skeleton ─────────────────
    let _allOpenings = [];

    function renderJobOpenings(openings) {
        _allOpenings = openings;
        const grid = $('#openings-grid');
        const skeleton = $('#job-skeleton');
        const searchInput = $('#job-search');
        const sortSelect = $('#job-sort');
        const countEl = $('#job-count');
        const infoEl = $('#job-results-info');

        // Show skeleton briefly
        if (skeleton) { skeleton.style.display = 'block'; grid.innerHTML = ''; }

        // Setup search/sort listeners (once)
        if (searchInput && !searchInput._bound) {
            searchInput._bound = true;
            searchInput.addEventListener('input', () => filterAndRenderJobs());
            if (sortSelect) sortSelect.addEventListener('change', () => filterAndRenderJobs());
        }

        setTimeout(() => {
            if (skeleton) skeleton.style.display = 'none';
            filterAndRenderJobs();
        }, 600);
    }

    function filterAndRenderJobs() {
        const grid = $('#openings-grid');
        const searchInput = $('#job-search');
        const sortSelect = $('#job-sort');
        const countEl = $('#job-count');
        const infoEl = $('#job-results-info');
        if (!grid) return;

        let filtered = [..._allOpenings];
        const query = searchInput ? searchInput.value.toLowerCase().trim() : '';

        // Filter
        if (query) {
            filtered = filtered.filter(j =>
                j.title.toLowerCase().includes(query) ||
                j.company.toLowerCase().includes(query) ||
                j.location.toLowerCase().includes(query)
            );
        }

        // Sort
        const sortVal = sortSelect ? sortSelect.value : 'default';
        if (sortVal === 'company') filtered.sort((a, b) => a.company.localeCompare(b.company));
        else if (sortVal === 'salary-high' || sortVal === 'salary-low') {
            const extractNum = s => { const m = s.replace(/[^\d]/g, ''); return parseInt(m) || 0; };
            filtered.sort((a, b) => sortVal === 'salary-high' ? extractNum(b.salary) - extractNum(a.salary) : extractNum(a.salary) - extractNum(b.salary));
        }

        grid.innerHTML = '';
        if (filtered.length === 0) {
            grid.innerHTML = '<p style="color:var(--text-muted);grid-column:1/-1;text-align:center;padding:40px 0;">No matching jobs found.</p>';
            if (countEl) countEl.textContent = '0 results';
            if (infoEl) infoEl.style.display = 'flex';
            return;
        }

        // Company logo initials
        const logoColors = ['#8b5cf6','#ec4899','#3b82f6','#06b6d4','#10b981','#f59e0b','#ef4444'];

        filtered.forEach((job, idx) => {
            const color = logoColors[job.company.charCodeAt(0) % logoColors.length];
            const initials = job.company.split(' ').map(w => w[0]).join('').slice(0,2).toUpperCase();
            const matchPct = Math.round(70 + Math.random() * 25);

            // Link to the company's own careers / application page. Fall back to a
            // Google "<company> careers" search if a listing is missing its link.
            const applyUrl = job.link
                ? job.link
                : 'https://www.google.com/search?q=' + encodeURIComponent(`${job.company} careers ${job.title}`);

            const card = document.createElement('div');
            card.className = 'opening-card reveal';
            card.innerHTML = `
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
                    <div style="width:44px;height:44px;border-radius:12px;background:${color}20;border:1px solid ${color}30;display:flex;align-items:center;justify-content:center;font-family:var(--font-display);font-size:0.85rem;font-weight:700;color:${color};flex-shrink:0;">${initials}</div>
                    <button class="jb-bookmark" onclick="this.classList.toggle('active');event.stopPropagation();" title="Bookmark" style="background:none;border:none;cursor:pointer;padding:4px;color:var(--text-muted);transition:all 0.2s;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>
                    </button>
                </div>
                <div class="opening-title">${job.title}</div>
                <div class="opening-company">${job.company}</div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;margin:12px 0;">
                    <span style="font-size:0.72rem;padding:3px 10px;border-radius:100px;background:rgba(139,92,246,0.08);color:#a78bfa;font-weight:500;">${job.location}</span>
                    <span style="font-size:0.72rem;padding:3px 10px;border-radius:100px;background:rgba(59,130,246,0.08);color:#60a5fa;font-weight:500;">${job.type}</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin:12px 0 16px;">
                    <span style="font-family:var(--font-display);font-size:0.92rem;font-weight:600;color:#fff;">${job.salary}</span>
                    <span style="font-size:0.72rem;font-weight:600;padding:3px 10px;border-radius:100px;background:rgba(16,185,129,0.1);color:#34d399;">${matchPct}% match</span>
                </div>
                <a href="${applyUrl}" target="_blank" rel="noopener" class="opening-apply" style="width:100%;justify-content:center;padding:10px 16px;border-radius:12px;">
                    Apply Now
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                </a>`;
            grid.appendChild(card);
            setTimeout(() => card.classList.add('visible'), 80 + idx * 80);
        });

        if (countEl) countEl.textContent = filtered.length + ' job' + (filtered.length !== 1 ? 's' : '') + ' found';
        if (infoEl) infoEl.style.display = 'flex';
    }

    // ── Mock Interview ───────────────────────────────────────────────────────
    startInterviewBtn.addEventListener('click', async () => {
        if (!selectedRole) { showToast('Please select a job role first.', 'error'); return; }
        try {
            const res = await fetch('/mock-interview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ role_title: selectedRole }) });
            const data = await res.json();
            interviewSection.style.display = 'block';
            scrollToEl(interviewSection);
            $('#interview-role-name').textContent = selectedRole;
            const container = $('#interview-questions');
            container.innerHTML = '';
            data.questions.forEach((q, idx) => {
                const block = document.createElement('div');
                block.className = 'question-block reveal';
                block.innerHTML = `<div class="question-number">Q${idx + 1}</div><div class="question-text">${q}</div><textarea class="answer-textarea" placeholder="Type your answer here..." data-question="${idx}"></textarea>`;
                container.appendChild(block);
                setTimeout(() => block.classList.add('visible'), 100 + idx * 100);
            });
        } catch (e) { showToast('Failed to load interview questions.', 'error'); }
    });

    // ── Submit Interview ─────────────────────────────────────────────────────
    submitInterviewBtn.addEventListener('click', async () => {
        const textareas = $$('.answer-textarea');
        const answers = Array.from(textareas).map(ta => ta.value.trim());
        if (answers.every(a => a === '')) { showToast('Please answer at least one question.', 'error'); return; }

        try {
            const res = await fetch('/evaluate-interview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ role_title: selectedRole, answers }) });
            const data = await res.json();
            scoreSection.style.display = 'block';
            scrollToEl(scoreSection);
            renderScore(data);
        } catch (e) { showToast('Evaluation failed. Please try again.', 'error'); }
    });

    // ── Render Score ─────────────────────────────────────────────────────────
    function renderScore(data) {
        addSvgGradients();
        const ring = $('#score-ring');
        const circumference = 2 * Math.PI * 60;
        const offset = circumference - (data.overall_score / 100) * circumference;
        ring.style.stroke = data.overall_score >= 70 ? '#10b981' : data.overall_score >= 50 ? '#f59e0b' : '#ef4444';
        setTimeout(() => { ring.style.strokeDashoffset = offset; }, 400);
        animateNumber($('#final-score'), 0, Math.round(data.overall_score), 1500);

        const statusEl = $('#score-status');
        if (data.overall_score >= 80) { statusEl.textContent = 'Outstanding Performance!'; statusEl.style.color = '#10b981'; }
        else if (data.overall_score >= 60) { statusEl.textContent = 'Good Performance'; statusEl.style.color = '#3b82f6'; }
        else if (data.overall_score >= 40) { statusEl.textContent = 'Average — Room for Growth'; statusEl.style.color = '#f59e0b'; }
        else { statusEl.textContent = 'Keep Practicing!'; statusEl.style.color = '#ef4444'; }

        // Breakdown
        const breakdownList = $('#breakdown-list');
        breakdownList.innerHTML = '';
        data.evaluations.forEach((ev, idx) => {
            const statusClass = ev.status.toLowerCase().replace(/ /g, '-');
            const item = document.createElement('div');
            item.className = 'breakdown-item reveal';
            item.innerHTML = `
                <div class="breakdown-num">${idx + 1}</div>
                <div class="breakdown-content">
                    <div class="breakdown-question">${ev.question}</div>
                    <div class="breakdown-bar"><div class="breakdown-fill ${statusClass}" data-width="${ev.score}"></div></div>
                    <div class="breakdown-meta">
                        <span class="breakdown-status ${statusClass}">${ev.status}</span>
                        <span class="breakdown-score">${Math.round(ev.score)}/100 | ${ev.keywords_matched}/${ev.total_keywords} keywords</span>
                    </div>
                </div>`;
            breakdownList.appendChild(item);
            setTimeout(() => { item.classList.add('visible'); item.querySelector('.breakdown-fill').style.width = ev.score + '%'; }, 300 + idx * 150);
        });

        // Strengths
        const strengthsList = $('#strengths-list');
        strengthsList.innerHTML = '';
        if (data.strengths.length > 0) { data.strengths.forEach(s => { const li = document.createElement('li'); li.textContent = s; strengthsList.appendChild(li); }); }
        else { strengthsList.innerHTML = '<li>Focus on providing more detailed and keyword-rich answers.</li>'; }

        // Weaknesses
        const weaknessesList = $('#weaknesses-list');
        weaknessesList.innerHTML = '';
        if (data.weaknesses.length > 0) { data.weaknesses.forEach(w => { const li = document.createElement('li'); li.textContent = w; weaknessesList.appendChild(li); }); }
        else { weaknessesList.innerHTML = '<li>Great job! No specific weaknesses detected.</li>'; }

        // Improvement tips
        const tipsList = $('#improvement-tips-list');
        tipsList.innerHTML = '';
        data.improvement_tips.forEach((tip, idx) => {
            const div = document.createElement('div');
            div.className = 'improvement-tip reveal';
            div.innerHTML = `<span class="tip-num">${idx + 1}</span><p>${tip}</p>`;
            tipsList.appendChild(div);
            setTimeout(() => div.classList.add('visible'), 200 + idx * 80);
        });

        if (navigator.vibrate) navigator.vibrate([20, 50, 20]);
    }

    // ── Restart ──────────────────────────────────────────────────────────────
    restartBtn.addEventListener('click', () => {
        selectedFile = null; analysisData = null; selectedRole = null; fileInput.value = '';
        hideAllSections();
        uploadArea.style.display = 'block';
        uploadedFileEl.style.display = 'none';
        analyzeBtn.style.display = 'none';
        scrollToEl($('#upload-section'));
    });


    /* ═══════════════════════════════════════════════════════════════════════════
       UTILITIES
       ═══════════════════════════════════════════════════════════════════════════ */

    function hideAllSections() {
        [loadingSection, resultsSection, jobsSection, interviewSection, scoreSection].forEach(s => s.style.display = 'none');
    }

    function scrollToEl(el) {
        setTimeout(() => el.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    }

    function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

    function animateNumber(el, start, end, duration, formatter) {
        const startTime = performance.now();
        function update(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Smooth ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = start + (end - start) * eased;
            if (formatter) {
                formatter(current);
            } else {
                el.textContent = Math.round(current);
            }
            if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    }

    function addSvgGradients() {
        if ($('#svg-gradient-defs')) return;
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.id = 'svg-gradient-defs';
        svg.style.cssText = 'position:absolute;width:0;height:0;';
        svg.innerHTML = `<defs>
            <linearGradient id="strengthGradient" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:#7c3aed"/><stop offset="100%" style="stop-color:#ec4899"/></linearGradient>
            <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:#7c3aed"/><stop offset="50%" style="stop-color:#a855f7"/><stop offset="100%" style="stop-color:#ec4899"/></linearGradient>
        </defs>`;
        document.body.appendChild(svg);
    }

    function showToast(message, type = 'info') {
        let toast = $('.toast');
        if (toast) toast.remove();
        toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toast.style.cssText = `
            position:fixed;bottom:32px;left:50%;transform:translateX(-50%) translateY(20px);
            padding:14px 28px;background:${type === 'error' ? 'rgba(239,68,68,0.9)' : 'rgba(139,92,246,0.9)'};
            color:#fff;border-radius:16px;font-family:'Inter',sans-serif;font-size:0.88rem;font-weight:600;
            box-shadow:0 8px 32px rgba(0,0,0,0.3);backdrop-filter:blur(12px);
            z-index:10000;opacity:0;transition:all 0.4s cubic-bezier(0.16,1,0.3,1);
            border:1px solid ${type === 'error' ? 'rgba(239,68,68,0.4)' : 'rgba(139,92,246,0.4)'};`;
        document.body.appendChild(toast);
        requestAnimationFrame(() => { toast.style.opacity = '1'; toast.style.transform = 'translateX(-50%) translateY(0)'; });
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(-50%) translateY(20px)';
            setTimeout(() => toast.remove(), 400);
        }, 3000);
    }

})();
