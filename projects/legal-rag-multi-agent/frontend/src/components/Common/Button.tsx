import { ButtonHTMLAttributes, forwardRef } from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '@/utils/cn';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  loading?: boolean;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, children, loading, variant = 'primary', size = 'md', disabled, ...props }, ref) => {
    const baseStyles =
      'inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-0 disabled:opacity-50 disabled:cursor-not-allowed';

    const variants = {
      primary:
        'text-zinc-950 bg-[var(--od-orange)] hover:bg-[var(--od-orange-strong)] shadow-[0_6px_18px_rgba(209,154,102,0.22)] hover:shadow-[0_8px_22px_rgba(209,154,102,0.28)] focus:ring-[var(--od-orange)]',
      secondary:
        'bg-[#3a404b] text-[var(--od-text-strong)] hover:bg-[#444b57] focus:ring-[#555d6b]',
      outline:
        'border border-[var(--od-border)] text-[var(--od-text-strong)] hover:bg-[#2f3540] focus:ring-[var(--od-orange)]',
      ghost:
        'text-[var(--od-text)] hover:bg-[#323846] focus:ring-[#4a5261]',
    };

    const sizes = {
      sm: 'px-3 py-1.5 text-sm',
      md: 'px-4 py-2',
      lg: 'px-6 py-3 text-lg',
    };

    return (
      <button
        ref={ref}
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        disabled={disabled || loading}
        {...props}
      >
        {loading && <Loader2 className="w-4 h-4 animate-spin" />}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';
