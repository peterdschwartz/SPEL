// function attachToggleListeners() {
//     const togglers = document.getElementsByClassName("toggler");
//     for (let i = 0; i < togglers.length; i++) {
//         togglers[i].addEventListener("click", function() {
//             const child = this.parentElement.querySelector(".child");
//             if (child) {
//                 child.classList.toggle("active");
//                 this.classList.toggle("active-toggler");
//             }
//         });
//     }
// }


// Helper used by both individual toggles and Toggle All
function setExpanded(togglerEl, expanded) {
    const child = togglerEl.nextElementSibling;
    if (!child || !child.classList.contains('child')) return;

    child.classList.toggle('active', expanded);
    togglerEl.classList.toggle('is-open', expanded);
    togglerEl.setAttribute('aria-expanded', expanded ? 'true' : 'false');
}

// Delegated handler for individual toggles (what you have now)
document.addEventListener('click', function(e) {
    const t = e.target.closest('.toggler');
    if (!t) return;
    setExpanded(t, !t.classList.contains('is-open'));
});

// Delegated handler for the Toggle All button
document.addEventListener('click', function(e) {
    const btn = e.target.closest('#toggle-all-button');
    if (!btn) return;

    // Scope to a container, if provided (e.g., #SubTree or .details-pane)
    const root = btn.dataset.target ? document.querySelector(btn.dataset.target) : document;

    // Only togglers that actually control a .child right next to them
    const togglers = [...root.querySelectorAll('.toggler')].filter(t => {
        const sib = t.nextElementSibling;
        return sib && sib.classList.contains('child');
    });

    if (togglers.length === 0) return;

    // If ANY is collapsed, expand all; else collapse all
    const anyCollapsed = togglers.some(t => !t.classList.contains('is-open'));
    const newState = anyCollapsed; // true = expand all, false = collapse all

    togglers.forEach(t => setExpanded(t, newState));

    // Optional: update button label and aria-pressed
    btn.textContent = newState ? 'Collapse all' : 'Expand all';
    btn.setAttribute('aria-pressed', newState ? 'true' : 'false');
});


// document.addEventListener("DOMContentLoaded", (event) => {
//     console.log("Attaching Toggle All Button")
//     const toggleAllButton = document.getElementById('toggle-all-button');
//     if (toggleAllButton) {
//         toggleAllButton.addEventListener('click', () => {
//             const children = document.querySelectorAll('.child');
//             const boxes = document.querySelectorAll('.box');
//             // make arrays
//             const any_active = Array.from(children).some(child => child.classList.contains('active'));
//             children.forEach(child => child.classList.toggle("active", !any_active))
//             boxes.forEach(box => box.classList.toggle("check-box", !any_active))
//         });
//     }
// })

// document.addEventListener("DOMContentLoaded", attachToggleListeners);
//
// // Attach listeners after HTMX updates (if HTMX is used)
// document.body.addEventListener('htmx:afterSwap', (evt) => {
//     attachToggleListeners(evt.detail.target || document);
// });
// document.body.addEventListener('htmx:afterSwap', () => {
//     attachToggleListeners();
// });

