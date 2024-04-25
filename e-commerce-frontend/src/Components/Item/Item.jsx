import React from 'react';
import './Item.css';
import { Link } from 'react-router-dom';

const Item = ({ BRAND, Title, Image_URL, id, new_price, old_price }) => {
  return (
    <div className='item'>
      <Link to={`/product/${id}`} style={{ textDecoration: 'none' }}>
        <img onClick={window.scrollTo(0, 0)} src={Image_URL} alt="products" />
      </Link>
      <p>{Title}</p>
      <div className="item-prices">
        <div className="item-price-new">${new_price}</div>
        <div className="item-price-old">${old_price}</div>
      </div>
    </div>
  );
};

export default Item;
