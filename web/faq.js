const search = document.querySelector('#faqSearch');
const groups = [...document.querySelectorAll('.faq-group')];
const items = [...document.querySelectorAll('.faq-item')];
const categoryButtons = [...document.querySelectorAll('#categoryNav button')];
const resultCount = document.querySelector('#resultCount');
const emptyState = document.querySelector('#emptyState');
let activeCategory = 'all';

function normalise(value = '') {
  return value.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
}

function applyFilters() {
  const query = normalise(search.value.trim());
  let visibleCount = 0;
  groups.forEach(group => {
    let groupCount = 0;
    group.querySelectorAll('.faq-item').forEach(item => {
      const matchesCategory = activeCategory === 'all' || group.dataset.group === activeCategory;
      const haystack = normalise(`${item.textContent} ${item.dataset.search || ''}`);
      const matchesSearch = !query || query.split(/\s+/).every(term => haystack.includes(term));
      const visible = matchesCategory && matchesSearch;
      item.classList.toggle('hidden', !visible);
      if (visible) groupCount += 1;
    });
    group.classList.toggle('hidden', groupCount === 0);
    visibleCount += groupCount;
  });
  resultCount.textContent = `Showing ${visibleCount} question${visibleCount === 1 ? '' : 's'}`;
  emptyState.classList.toggle('hidden', visibleCount !== 0);
}

categoryButtons.forEach(button => button.addEventListener('click', () => {
  activeCategory = button.dataset.category;
  categoryButtons.forEach(item => item.classList.toggle('active', item === button));
  applyFilters();
  document.querySelector('#questions-title').scrollIntoView({behavior: 'smooth', block: 'start'});
}));

search.addEventListener('input', applyFilters);
document.querySelector('#clearSearch').addEventListener('click', () => {
  search.value = '';
  activeCategory = 'all';
  categoryButtons.forEach(button => button.classList.toggle('active', button.dataset.category === 'all'));
  applyFilters();
  search.focus();
});

document.addEventListener('keydown', event => {
  if (event.key === '/' && document.activeElement !== search) {
    event.preventDefault();
    search.focus();
  }
  if (event.key === 'Escape' && document.activeElement === search) {
    search.value = '';
    applyFilters();
    search.blur();
  }
});

function openHashTarget() {
  if (!location.hash) return;
  const target = document.querySelector(location.hash);
  if (target?.matches('details')) {
    target.open = true;
    requestAnimationFrame(() => target.scrollIntoView({behavior: 'smooth', block: 'center'}));
  }
}

openHashTarget();
window.addEventListener('hashchange', openHashTarget);
