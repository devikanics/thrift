import React from 'react';
import './Item.css';
import { Link } from 'react-router-dom';
import axios from 'axios';

const Item = (props) => {

  const handleClick = () => {
    const id = props.id;
    sendIdToServer(id);
  };

  const sendIdToServer = (id) => {
    axios.post('http://localhost:4000/product-info', { id }, {
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': 'http://localhost:3000'
      }
    })
    .then(response => {
      // Handle response if needed
    })
    .catch(error => {
      // Handle error
      console.error('Error sending product ID:', error);
    });
  };

  return (
    <div className='item'>
      <Link to={`/product/${props.id}`} style={{ textDecoration: 'none' }}>
        <img onClick={handleClick} src={props.image} alt="products" />
      </Link>
      <p>{props.name}</p>
      <div className="item-prices">
        <div className="item-price-new">${props.new_price}</div>
        <div className="item-price-old">${props.old_price}</div>
      </div>
    </div>
  );
};

export default Item;