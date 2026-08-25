const artifacts = [...document.querySelectorAll(".artifact")];

async function loadSource(artifact) {
  if (artifact.dataset.loaded === "true") return;

  const code = artifact.querySelector("code");

  if (window.location.protocol === "file:") {
    const frame = document.createElement("iframe");
    frame.className = "source-frame";
    frame.src = artifact.dataset.source;
    frame.title = `${artifact.querySelector(".artifact-name").textContent} full source`;
    code.closest("pre").replaceWith(frame);
    artifact.dataset.loaded = "true";
    return;
  }

  try {
    const response = await fetch(artifact.dataset.source);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    code.textContent = await response.text();
    window.hljs?.highlightElement(code);
    artifact.dataset.loaded = "true";
  } catch {
    code.textContent =
      "The embedded preview could not load. Open the raw file above to view the complete source.";
  }
}

artifacts.forEach((artifact) => {
  artifact.addEventListener("toggle", () => {
    if (!artifact.open) return;

    artifacts.forEach((other) => {
      if (other !== artifact) other.open = false;
    });

    loadSource(artifact);
  });
});
