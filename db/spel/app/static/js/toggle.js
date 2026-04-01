function getChildByAria(toggler) {
  const id = toggler.getAttribute('aria-controls');
  return id ? document.getElementById(id) : null;
}

function setExpanded(toggler, expanded) {
  const child = getChildByAria(toggler);
  const hasChild = !!child;

  // Only flip the child when it exists
  if (hasChild) child.classList.toggle('active', !!expanded);

  // Sync visual state strictly to actual result
  const isOpen = hasChild && !!expanded;
  toggler.classList.toggle('is-open', isOpen);
  toggler.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
}

document.addEventListener('click', (e) => {
  const t = e.target.closest('.toggler[role="button"]');
  if (!t) return;
  setExpanded(t, t.getAttribute('aria-expanded') !== 'true');
});

document.addEventListener('keydown', (e) => {
  const t = e.target.closest('.toggler[role="button"]');
  if (!t) return;
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    setExpanded(t, t.getAttribute('aria-expanded') !== 'true');
  }
});

document.addEventListener('click', (e) => {
  const btn = e.target.closest('[data-toggle-all]');
  if (!btn) return;

  const root = btn.dataset.target ? document.querySelector(btn.dataset.target) : document;
  const togglers = Array
    .from(root.querySelectorAll('.toggler[role="button"][aria-controls]'))
    .filter(t => getChildByAria(t)); // must actually control something

  if (!togglers.length) return;

  // If ANY child is closed, we'll open everything; else close all
  const anyClosed = togglers.some(t => !getChildByAria(t).classList.contains('active'));
  const newState = anyClosed;

  togglers.forEach(t => setExpanded(t, newState));

  btn.textContent = newState ? 'Collapse all' : 'Expand all';
  btn.setAttribute('aria-pressed', newState ? 'true' : 'false');
});

function syncTogglers(root = document) {
  const togglers = root.querySelectorAll('.toggler[role="button"][aria-controls]');
  togglers.forEach(t => {
    const child = getChildByAria(t);
    const isOpen = !!child && child.classList.contains('active');
    t.classList.toggle('is-open', isOpen);
    t.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
  });
}

document.addEventListener('DOMContentLoaded', () => syncTogglers());
document.body.addEventListener('htmx:afterSwap', (e) => {
  // scope to the swapped fragment
  syncTogglers(e.target);
});
