import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

const SearchResultsPage = () => {
  const location = useLocation();
  const searchQuery = new URLSearchParams(location.search).get('query');

  useEffect(() => {
    // Perform any necessary actions based on the search query
    console.log('Search query:', searchQuery);
  }, [searchQuery]);

  return (
    <div>
      <h1>Search Results</h1>
      <p>Display search results here</p>
    </div>
  );
}

export default SearchResultsPage;