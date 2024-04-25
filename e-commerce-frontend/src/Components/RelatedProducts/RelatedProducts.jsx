import React from 'react';
import './RelatedProducts.css';
import Item from '../Item/Item';
import recom from '../../rs';

const RelatedProducts = () => {
  console.log(recom)
  return (
    <div className='relatedproducts'>
      <h1>Related Products</h1>
      <hr />
      <div className="relatedproducts-items">
        {recom.map((item, index) => (
          <Item
            className="item"
            key={index} // Adding a unique key for each item
            BRAND={item.BRAND}
            Title={item.Title}
            Image_URL={item.Image_URL}
          />
        ))}
      </div>
    </div>
  );
};

export default RelatedProducts;
