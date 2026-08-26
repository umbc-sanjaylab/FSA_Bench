
const menuToggle = document.querySelector(".menu-toggle");
const navLinks = document.querySelector(".nav-links");

if (menuToggle) {
  menuToggle.addEventListener("click", () => {
    navLinks.classList.toggle("open");
  });
}

document.querySelectorAll(".nav-links a").forEach(link => {
  link.addEventListener("click", () => navLinks.classList.remove("open"));
});

const copyButton = document.getElementById("copyCitation");
if (copyButton) {
  copyButton.addEventListener("click", async () => {
    const citation = `@inproceedings{ahmed2026fsabench,
  title     = {FSA-Bench: Benchmarking Federated Survival Models},
  author    = {Ahmed, Sultan and Purushotham, Sanjay},
  booktitle = {Proceedings of Machine Learning Research},
  year      = {2026}
}`;
    try {
      await navigator.clipboard.writeText(citation);
      copyButton.textContent = "Copied ✓";
      setTimeout(() => copyButton.textContent = "Copy citation", 1600);
    } catch {
      copyButton.textContent = "Copy unavailable";
    }
  });
}
