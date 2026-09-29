(function () {
  const SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" width="90%" height="90%" viewBox="0 0 24 24" aria-hidden="true"><path fill="#ff4444" d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>';

  function clickYtTrigger() {
    const root = document.getElementById("yt-source-trigger");
    if (!root) return;
    const btn = root.matches("button") ? root : root.querySelector("button");
    btn?.click();
  }

  function attachYoutubeTab() {
    const stack = document.getElementById("input-audio-stack");
    if (!stack) return;

    const sel =
      stack.querySelector('[data-testid="source-select"]') ||
      stack.querySelector(".source-selection");
    if (!sel) return;

    let ytTab = sel.querySelector(".resemble-yt-tab");
    if (!ytTab) {
      ytTab = document.createElement("button");
      ytTab.type = "button";
      ytTab.className = "icon resemble-yt-tab svelte-exvkcd";
      ytTab.setAttribute("aria-label", "YouTube");
      ytTab.title = "YouTube";
      ytTab.innerHTML = SVG;
      ytTab.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        openYoutubePanel();
        clickYtTrigger();
      });
      sel.appendChild(ytTab);
    }

    if (!sel.dataset.ytBound) {
      sel.dataset.ytBound = "1";
      sel.addEventListener("click", (ev) => {
        const tab = ev.target.closest("button.icon");
        if (tab && !tab.classList.contains("resemble-yt-tab")) {
          closeYoutubePanel();
          document.getElementById("close-yt-mode")?.querySelector("button")?.click();
        }
      });
    }
  }

  function openYoutubePanel() {
    attachYoutubeTab();
    document.getElementById("input-audio-stack")?.classList.add("yt-open");
    const sel = document
      .getElementById("input-audio-stack")
      ?.querySelector('[data-testid="source-select"]');
    const ytTab = sel?.querySelector(".resemble-yt-tab");
    sel?.querySelectorAll("button.icon").forEach((b) => b.classList.remove("selected"));
    ytTab?.classList.add("selected");
  }

  function closeYoutubePanel() {
    document.getElementById("input-audio-stack")?.classList.remove("yt-open");
    const sel = document
      .getElementById("input-audio-stack")
      ?.querySelector('[data-testid="source-select"]');
    sel?.querySelector(".resemble-yt-tab")?.classList.remove("selected");
    sel?.querySelector("button.icon")?.classList.add("selected");
  }

  window.resembleAttachYoutubeTab = attachYoutubeTab;
  window.resembleOpenYoutube = openYoutubePanel;
  window.resembleCloseYoutube = closeYoutubePanel;

  attachYoutubeTab();
  document.addEventListener("DOMContentLoaded", attachYoutubeTab);
  new MutationObserver(attachYoutubeTab).observe(document.documentElement, {
    childList: true,
    subtree: true,
  });
  setInterval(attachYoutubeTab, 1500);
})();
