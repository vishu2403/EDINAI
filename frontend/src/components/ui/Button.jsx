/**
 * Button Component
 * Reusable button with various styles matching INAI design
 */

import React from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';

const Button = ({
  children,
  variant = 'primary',
  size = 'medium',
  loading = false,
  disabled = false,
  fullWidth = false,
  icon: Icon = null,
  iconPosition = 'left',
  className = '',
  onClick,
  type = 'button',
  ...props
}) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium rounded-xl transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-dark-bg';
  
  const variants = {
    primary: 'bg-primary-600 hover:bg-primary-700 text-white shadow-lg hover:shadow-glow focus:ring-primary-500',
    secondary: 'bg-dark-card hover:bg-gray-700 text-dark-text-primary border border-dark-border hover:border-gray-600 focus:ring-gray-500',
    danger: 'bg-red-600 hover:bg-red-700 text-white shadow-lg hover:shadow-red-500/20 focus:ring-red-500',
    ghost: 'text-dark-text-secondary hover:text-dark-text-primary hover:bg-dark-card focus:ring-gray-500',
    success: 'bg-green-600 hover:bg-green-700 text-white shadow-lg hover:shadow-green-500/20 focus:ring-green-500',
    warning: 'bg-yellow-600 hover:bg-yellow-700 text-white shadow-lg hover:shadow-yellow-500/20 focus:ring-yellow-500',
    gradient: 'bg-gradient-to-r from-primary-600 to-blue-600 hover:from-primary-700 hover:to-blue-700 text-white shadow-lg hover:shadow-glow focus:ring-primary-500',
  };

  const sizes = {
    small: 'px-3 py-1.5 text-sm',
    medium: 'px-4 py-2.5 text-sm',
    large: 'px-6 py-3 text-base',
    xlarge: 'px-8 py-4 text-lg',
  };

  const iconSizes = {
    small: 'w-4 h-4',
    medium: 'w-5 h-5',
    large: 'w-5 h-5',
    xlarge: 'w-6 h-6',
  };

  const isDisabled = disabled || loading;

  const classes = clsx(
    baseClasses,
    variants[variant],
    sizes[size],
    {
      'w-full': fullWidth,
      'opacity-50 cursor-not-allowed': isDisabled,
      'cursor-pointer': !isDisabled,
    },
    className
  );

  const handleClick = (e) => {
    if (!isDisabled && onClick) {
      onClick(e);
    }
  };

  return (
    <motion.button
      type={type}
      className={classes}
      onClick={handleClick}
      disabled={isDisabled}
      whileHover={!isDisabled ? { scale: 1.02 } : {}}
      whileTap={!isDisabled ? { scale: 0.98 } : {}}
      {...props}
    >
      {loading && (
        <div className={clsx('spinner mr-2', iconSizes[size])} />
      )}
      
      {!loading && Icon && iconPosition === 'left' && (
        <Icon className={clsx(iconSizes[size], children ? 'mr-2' : '')} />
      )}
      
      {children}
      
      {!loading && Icon && iconPosition === 'right' && (
        <Icon className={clsx(iconSizes[size], children ? 'ml-2' : '')} />
      )}
    </motion.button>
  );
};

export default Button;
