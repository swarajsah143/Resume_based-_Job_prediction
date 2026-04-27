/* ════════════════════════════════════════════════════════════════════════════
   ResumeAI — Dark Premium Frontend Logic
   Micro-interactions, ripple effects, scroll animations, haptic feedback
   ════════════════════════════════════════════════════════════════════════════ */

(() => {
    'use strict';

    // ─── State ──────────────────────────────────────────────────────────────
    let selectedFile = null;
    let analysisData = null;
    let selectedRole = null;

    // ─── DOM Elements ───────────────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const uploadArea = $('#upload-area');
    const fileInput = $('#file-input');
    const uploadedFileEl = $('#uploaded-file');
    const fileNameEl = $('#file-name');
    const removeFileBtn = $('#remove-file');
    const analyzeBtn = $('#analyze-btn');
    const loadingSection = $('#loading-section');
    const loadingText = $('#loading-text');
    const loadingBarFill = $('#loading-bar-fill');
    const resultsSection = $('#results-section');
    const jobsSection = $('#jobs-section');
    const interviewSection = $('#interview-section');
    const scoreSection = $('#score-section');
    const startInterviewBtn = $('#start-interview-btn');
    const submitInterviewBtn = $('#submit-interview-btn');
    const restartBtn = $('#restart-btn');

    // ─── Ripple Effect ──────────────────────────────────────────────────────
    function createRipple(e) {
        const btn = e.currentTarget;
        const circle = document.createElement('span');
        const diameter = Math.max(btn.clientWidth, btn.clientHeight);
        const radius = diameter / 2;
        const rect = btn.getBoundingClientRect();
        circle.style.width = circle.style.height = diameter + 'px';
        circle.style.left = (e.clientX - rect.left - radius) + 'px';
        circle.style.top = (e.clientY - rect.top - radius) + 'px';
        circle.className = 'ripple';
        const oldRipple = btn.querySelector('.ripple');
        if (oldRipple) oldRipple.remove();
        btn.appendChild(circle);
        // Haptic feedback for mobile
        if (navigator.vibrate) navigator.vibrate(10);
    }

    // Attach ripple to all buttons
    function attachRipples() {
        document.querySelectorAll('.btn').forEach(btn => {
            btn.removeEventListener('click', createRipple);
            btn.addEventListener('click', createRipple);
        });
    }
    attachRipples();

    // ─── Scroll Reveal ──────────────────────────────────────────────────────
    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

    function initReveal() {
        document.querySelectorAll('.feature-card, .step, .result-card, .tip-card, .opening-card, .role-card, .question-block, .breakdown-item').forEach((el, i) => {
            el.classList.add('reveal');
            el.style.transitionDelay = (i % 6) * 80 + 'ms';
            revealObserver.observe(el);
        });
    }
    initReveal();

    // ─── Navbar Scroll ──────────────────────────────────────────────────────
    window.addEventListener('scroll', () => {
        const navbar = $('#navbar');
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // ─── Smooth Scroll ──────────────────────────────────────────────────────
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });

    // ─── Upload Handling ────────────────────────────────────────────────────
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

    // ─── Analyze Resume ─────────────────────────────────────────────────────
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;
        hideAllSections();
        loadingSection.style.display = 'block';
        scrollTo(loadingSection);
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
        const messages = ['Extracting text from resume...', 'Identifying skills and technologies...', 'Analyzing education and experience...', 'Matching with job roles...', 'Generating personalized insights...'];
        let progress = 0;
        const interval = setInterval(() => {
            progress += 5;
            loadingBarFill.style.width = Math.min(progress, 95) + '%';
            const msgIdx = Math.floor((progress / 100) * messages.length);
            loadingText.textContent = messages[Math.min(msgIdx, messages.length - 1)];
            if (progress >= 95) clearInterval(interval);
        }, 100);
    }

    // ─── Show Results ───────────────────────────────────────────────────────
    function showResults(data) {
        resultsSection.style.display = 'block';
        scrollTo(resultsSection);
        $('#result-filename').textContent = data.filename;
        animateStrength(data.resume_strength);

        // Skills
        const skillTags = $('#skill-tags');
        skillTags.innerHTML = '';
        if (data.skills.length > 0) {
            data.skills.forEach(skill => {
                const tag = document.createElement('span');
                tag.className = 'skill-tag';
                tag.textContent = skill.name;
                tag.title = skill.category;
                skillTags.appendChild(tag);
            });
        } else {
            skillTags.innerHTML = '<span style="color:var(--text-muted);font-size:0.88rem;">No skills detected. Try uploading a more detailed resume.</span>';
        }

        // Education
        const eduList = $('#education-list');
        eduList.innerHTML = '';
        if (data.education.length > 0) {
            data.education.forEach(item => { const li = document.createElement('li'); li.textContent = item; eduList.appendChild(li); });
        } else {
            eduList.innerHTML = '<li style="list-style:none;padding-left:0;">No education details detected.</li>';
        }

        // Experience
        const expList = $('#experience-list');
        expList.innerHTML = '';
        if (data.experience.length > 0) {
            data.experience.forEach(item => { const li = document.createElement('li'); li.textContent = item; expList.appendChild(li); });
        } else {
            expList.innerHTML = '<li style="list-style:none;padding-left:0;">No experience details detected.</li>';
        }

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

        // Re-attach ripples and reveal for dynamic cards
        attachRipples();
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

        if (score >= 80) { statusEl.textContent = '🌟 Excellent Resume'; statusEl.style.color = '#10b981'; descEl.textContent = 'Your resume is well-structured with strong skills and experience.'; }
        else if (score >= 60) { statusEl.textContent = '✅ Good Resume'; statusEl.style.color = '#3b82f6'; descEl.textContent = 'Covers the basics well. Adding projects and certifications can boost it.'; }
        else if (score >= 40) { statusEl.textContent = '⚡ Needs Improvement'; statusEl.style.color = '#f59e0b'; descEl.textContent = 'Room to grow. Add more skills, experience details, and keywords.'; }
        else { statusEl.textContent = '⚠️ Weak Resume'; statusEl.style.color = '#ef4444'; descEl.textContent = 'Needs significant improvements. Focus on skills, projects, and summary.'; }
    }

    // ─── Select Role ────────────────────────────────────────────────────────
    async function selectRole(roleTitle, cardEl) {
        selectedRole = roleTitle;
        $$('.role-card').forEach(c => c.classList.remove('selected'));
        cardEl.classList.add('selected');
        if (navigator.vibrate) navigator.vibrate(10);

        jobsSection.style.display = 'block';
        scrollTo(jobsSection);
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
                <div class="gap-tags">${data.missing.map(s => `<span class="gap-tag missing">${s}</span>`).join('')}${data.missing.length === 0 ? '<span style="color:var(--green);font-size:0.85rem;">You have all required skills! 🎉</span>' : ''}</div>
            </div>`;
    }

    function renderJobOpenings(openings) {
        const grid = $('#openings-grid');
        grid.innerHTML = '';
        if (openings.length === 0) { grid.innerHTML = '<p style="color:var(--text-muted);grid-column:1/-1;text-align:center;">No openings available for this role.</p>'; return; }
        openings.forEach((job, idx) => {
            const card = document.createElement('div');
            card.className = 'opening-card reveal';
            card.innerHTML = `
                <div class="opening-title">${job.title}</div>
                <div class="opening-company">${job.company}</div>
                <div class="opening-meta"><span>📍 ${job.location}</span><span>💼 ${job.type}</span><span>💰 ${job.salary}</span></div>
                <a href="${job.link}" target="_blank" rel="noopener" class="opening-apply">Apply Now <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></a>`;
            grid.appendChild(card);
            setTimeout(() => card.classList.add('visible'), 100 + idx * 100);
        });
    }

    // ─── Mock Interview ─────────────────────────────────────────────────────
    startInterviewBtn.addEventListener('click', async () => {
        if (!selectedRole) { showToast('Please select a job role first.', 'error'); return; }
        try {
            const res = await fetch('/mock-interview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ role_title: selectedRole }) });
            const data = await res.json();
            interviewSection.style.display = 'block';
            scrollTo(interviewSection);
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

    // ─── Submit Interview ───────────────────────────────────────────────────
    submitInterviewBtn.addEventListener('click', async () => {
        const textareas = $$('.answer-textarea');
        const answers = Array.from(textareas).map(ta => ta.value.trim());
        if (answers.every(a => a === '')) { showToast('Please answer at least one question.', 'error'); return; }

        try {
            const res = await fetch('/evaluate-interview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ role_title: selectedRole, answers }) });
            const data = await res.json();
            scoreSection.style.display = 'block';
            scrollTo(scoreSection);
            renderScore(data);
        } catch (e) { showToast('Evaluation failed. Please try again.', 'error'); }
    });

    // ─── Render Score ───────────────────────────────────────────────────────
    function renderScore(data) {
        addSvgGradients();
        const ring = $('#score-ring');
        const circumference = 2 * Math.PI * 60;
        const offset = circumference - (data.overall_score / 100) * circumference;
        ring.style.stroke = data.overall_score >= 70 ? '#10b981' : data.overall_score >= 50 ? '#f59e0b' : '#ef4444';
        setTimeout(() => { ring.style.strokeDashoffset = offset; }, 400);
        animateNumber($('#final-score'), 0, Math.round(data.overall_score), 1500);

        const statusEl = $('#score-status');
        if (data.overall_score >= 80) { statusEl.textContent = '🌟 Outstanding Performance!'; statusEl.style.color = '#10b981'; }
        else if (data.overall_score >= 60) { statusEl.textContent = '✅ Good Performance'; statusEl.style.color = '#3b82f6'; }
        else if (data.overall_score >= 40) { statusEl.textContent = '⚡ Average — Room for Growth'; statusEl.style.color = '#f59e0b'; }
        else { statusEl.textContent = '📚 Keep Practicing!'; statusEl.style.color = '#ef4444'; }

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
                        <span class="breakdown-score">${Math.round(ev.score)}/100 • ${ev.keywords_matched}/${ev.total_keywords} keywords</span>
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

    // ─── Restart ────────────────────────────────────────────────────────────
    restartBtn.addEventListener('click', () => {
        selectedFile = null; analysisData = null; selectedRole = null; fileInput.value = '';
        hideAllSections();
        uploadArea.style.display = 'block';
        uploadedFileEl.style.display = 'none';
        analyzeBtn.style.display = 'none';
        scrollTo($('#upload-section'));
    });

    // ─── Utilities ──────────────────────────────────────────────────────────
    function hideAllSections() {
        [loadingSection, resultsSection, jobsSection, interviewSection, scoreSection].forEach(s => s.style.display = 'none');
    }

    function scrollTo(el) {
        setTimeout(() => el.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    }

    function delay(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

    function animateNumber(el, start, end, duration) {
        const startTime = performance.now();
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = Math.round(start + (end - start) * eased);
            if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    }

    function addSvgGradients() {
        if (document.querySelector('#svg-gradient-defs')) return;
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.id = 'svg-gradient-defs';
        svg.style.cssText = 'position:absolute;width:0;height:0;';
        svg.innerHTML = `<defs>
            <linearGradient id="strengthGradient" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:#6366f1"/><stop offset="100%" style="stop-color:#a855f7"/></linearGradient>
            <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:#6366f1"/><stop offset="50%" style="stop-color:#a855f7"/><stop offset="100%" style="stop-color:#ec4899"/></linearGradient>
        </defs>`;
        document.body.appendChild(svg);
    }

    function showToast(message, type = 'info') {
        let toast = document.querySelector('.toast');
        if (toast) toast.remove();
        toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toast.style.cssText = `
            position:fixed; bottom:32px; left:50%; transform:translateX(-50%) translateY(20px);
            padding:12px 24px; background:${type === 'error' ? 'rgba(239,68,68,0.9)' : 'rgba(99,102,241,0.9)'};
            color:#fff; border-radius:12px; font-family:'Inter',sans-serif; font-size:0.9rem; font-weight:600;
            box-shadow:0 8px 32px rgba(0,0,0,0.3); backdrop-filter:blur(10px);
            z-index:10000; opacity:0; transition:all 0.3s ease;
            border:1px solid ${type === 'error' ? 'rgba(239,68,68,0.5)' : 'rgba(99,102,241,0.5)'};`;
        document.body.appendChild(toast);
        requestAnimationFrame(() => { toast.style.opacity = '1'; toast.style.transform = 'translateX(-50%) translateY(0)'; });
        setTimeout(() => { toast.style.opacity = '0'; toast.style.transform = 'translateX(-50%) translateY(20px)'; setTimeout(() => toast.remove(), 300); }, 3000);
    }

})();
