/**
 * Input Field Component
 * Reusable input field with validation and INAI styling
 */

import React, { forwardRef } from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { Eye, EyeOff } from 'lucide-react';

const InputField = forwardRef(({
  label,
  type = 'text',
  placeholder,
  value,
  onChange,
  onBlur,
  error,
  required = false,
  disabled = false,
  icon: Icon = null,
  showPasswordToggle = false,
  className = '',
  containerClassName = '',
  helperText = '',
  ...props
}, ref) => {
  const [showPassword, setShowPassword] = React.useState(false);
  const [isFocused, setIsFocused] = React.useState(false);

  const inputType = type === 'password' && showPassword ? 'text' : type;

  const handlePasswordToggle = () => {
    setShowPassword(!showPassword);
  };

  const inputClasses = clsx(
    'input-primary',
    {
      'input-error': error,
      'pl-12': Icon,
      'pr-12': showPasswordToggle || type === 'password',
      'opacity-50 cursor-not-allowed': disabled,
    },
    className
  );

  return (
    <motion.div
      className={clsx('space-y-2', containerClassName)}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {label && (
        <label className="block text-sm font-medium text-dark-text-primary">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      
      <div className="relative">
        {Icon && (
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Icon className="h-5 w-5 text-dark-text-muted" />
          </div>
        )}
        
        <input
          ref={ref}
          type={inputType}
          value={value}
          onChange={onChange}
          onBlur={(e) => {
            setIsFocused(false);
            if (onBlur) onBlur(e);
          }}
          onFocus={() => setIsFocused(true)}
          placeholder={placeholder}
          disabled={disabled}
          required={required}
          className={inputClasses}
          {...props}
        />
        
        {(showPasswordToggle || type === 'password') && (
          <button
            type="button"
            onClick={handlePasswordToggle}
            className="absolute inset-y-0 right-0 pr-3 flex items-center text-dark-text-muted hover:text-dark-text-secondary transition-colors"
            tabIndex={-1}
          >
            {showPassword ? (
              <EyeOff className="h-5 w-5" />
            ) : (
              <Eye className="h-5 w-5" />
            )}
          </button>
        )}
        
        {/* Focus indicator */}
        {isFocused && (
          <motion.div
            className="absolute inset-0 rounded-xl border-2 border-primary-500 pointer-events-none"
            initial={{ opacity: 0, scale: 1.02 }}
            animate={{ opacity: 0.3, scale: 1 }}
            exit={{ opacity: 0, scale: 1.02 }}
            transition={{ duration: 0.2 }}
          />
        )}
      </div>
      
      {error && (
        <motion.p
          className="text-sm text-red-500"
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2 }}
        >
          {error}
        </motion.p>
      )}
      
      {helperText && !error && (
        <p className="text-sm text-dark-text-muted">
          {helperText}
        </p>
      )}
    </motion.div>
  );
});

InputField.displayName = 'InputField';

export default InputField;
