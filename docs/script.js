"use strict";

const header = document.querySelector("[data-header]");
const navLinks = [...document.querySelectorAll('.site-nav a[href^="#"]')];
const sections = navLinks.map((link) => document.querySelector(link.getAttribute("href"))).filter(Boolean);
const updateHeader = () => {
  header?.classList.toggle("is-scrolled", window.scrollY > 12);
  if (window.scrollY < 24) {
    navLinks.forEach((link) => {
      link.classList.remove("is-active");
      link.removeAttribute("aria-current");
    });
  }
};
updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    navLinks.forEach((link) => {
      const active = link.getAttribute("href") === `#${visible.target.id}`;
      link.classList.toggle("is-active", active);
      if (active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }, { rootMargin: "-15% 0px -55% 0px", threshold: [0, 0.1, 0.3] });
  sections.forEach((section) => observer.observe(section));
}

const videos = [...document.querySelectorAll("[data-demo-video]")];
videos.forEach((video) => {
  video.addEventListener("play", () => videos.forEach((other) => { if (other !== video) other.pause(); }));
});

const rrVideo = document.querySelector("#rr-video");
const angleButtons = [...document.querySelectorAll("[data-angle]")];
angleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (button.getAttribute("aria-pressed") === "true") return;
    const angle = button.dataset.angle;
    rrVideo.pause();
    rrVideo.querySelector("source").src = `assets/rr-${angle}.mp4`;
    rrVideo.poster = `assets/rr-${angle}-poster.webp`;
    rrVideo.setAttribute("aria-label", `Real-world rectangular reorientation toward ${angle} degrees`);
    rrVideo.load();
    document.querySelector("#rr-caption").textContent = `Target heading: ${angle}° · Real-world trial`;
    document.querySelector("#rr-download").href = `assets/rr-${angle}.mp4`;
    angleButtons.forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
  });
});

// Clipboard API is preferred; selection-based copying supports local HTTP previews.
async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try { await navigator.clipboard.writeText(text); return; } catch { /* Try the local fallback. */ }
  }
  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  field.style.position = "fixed";
  field.style.opacity = "0";
  document.body.append(field);
  field.select();
  const copied = document.execCommand("copy");
  field.remove();
  if (!copied) throw new Error("Clipboard unavailable");
}

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const status = document.querySelector("#copy-status");
    try {
      await copyText(document.getElementById(button.dataset.copy).textContent.trim());
      button.textContent = "Copied ✓";
      status.textContent = "Installation commands copied.";
      window.setTimeout(() => { button.textContent = "Copy commands"; }, 2200);
    } catch {
      status.textContent = "Select the commands in the terminal block to copy them manually.";
    }
    button.focus({ preventScroll: true });
  });
});
