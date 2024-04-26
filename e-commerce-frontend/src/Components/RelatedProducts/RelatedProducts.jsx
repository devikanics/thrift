import React from 'react';
import './RelatedProducts.css';
import Item from '../Item/Item';
import recom from '../../rs';

const RelatedProducts = () => {
  return (
    <div className='relatedproducts'>

      <h1>Related Products</h1>
      <hr />
      <div className="relatedproducts-items">
        {recom.map((item, index) => (
          <Item
          className="item"
          key={index}
          id={item.id} 
          image={item.Image_URL} 
          name={item.Title} 
          new_price={item.new_price} 
          old_price={item.old_price} 
        />
        ))}
      </div>
    </div>
  );
};

export default RelatedProducts;
