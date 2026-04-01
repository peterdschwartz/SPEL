document.getElementById('Variable').addEventListener('input', function() {
    const query = this.value;
    const suggestionsList = document.getElementById('suggestions-list');

    if (query.length > 0) {
        // Send an AJAX request to get suggestions
        fetch(`/autocomplete/?q=${query}`)
            .then(response => response.json())
            .then(data => {
                suggestionsList.innerHTML = ''; // Clear previous suggestions
                data.forEach(item => {
                    const li = document.createElement('li');
                    li.textContent = item.name;
                    li.addEventListener('click', () => {
                        document.getElementById('Variable').value = item.name;
                        suggestionsList.innerHTML = ''; // Clear suggestions after selection
                    });
                    suggestionsList.appendChild(li);
                });
            });
    } else {
        suggestionsList.innerHTML = ''; // Clear suggestions if input is empty
    }
});

