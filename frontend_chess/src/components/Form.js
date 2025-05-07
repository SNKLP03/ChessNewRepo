import React from 'react';
import './css/Form.css';

const Form = ({ fields, onSubmit, buttonText, footerText, footerLink }) => {
  return (
    <div className="form-container">
      <form onSubmit={onSubmit}>
        {fields.map((field, index) => (
          <div className="form-group" key={index}>
            <label htmlFor={field.name}>{field.label}</label>
            <input
              type={field.type}
              id={field.name}
              name={field.name}
              placeholder={field.placeholder}
              aria-label={field.label}
            />
          </div>
        ))}
        <button type="submit" className="submit-btn">
          {buttonText}
        </button>
      </form>
      <p className="footer-text">
        {footerText} <a href={footerLink.href}>{footerLink.text}</a>
      </p>
    </div>
  );
};

export default Form;