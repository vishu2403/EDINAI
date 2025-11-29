import React, { useMemo } from 'react';
import { MathJax } from 'better-react-mathjax';

const formatInlineMath = (text) => {
  if (!text) return [];
  
  const parts = [];
  let currentPart = '';
  let inMath = false;
  
  for (let i = 0; i < text.length; i++) {
    // Handle escaped dollar signs
    if (text[i] === '\\' && text[i+1] === '$') {
      currentPart += '$';
      i++;
      continue;
    }
    
    // Toggle math mode on unescaped $
    if (text[i] === '$' && (i === 0 || text[i-1] !== '\\')) {
      if (currentPart) {
        parts.push({
          type: inMath ? 'math' : 'text',
          content: currentPart.trim()
        });
        currentPart = '';
      }
      inMath = !inMath;
    } else {
      currentPart += text[i];
    }
  }
  
  // Add the last part if it exists
  if (currentPart) {
    parts.push({
      type: inMath ? 'math' : 'text',
      content: currentPart.trim()
    });
  }
  
  return parts;
};

const MathText = ({ children }) => {
  const processedContent = useMemo(() => {
    if (!children) return null;
    if (typeof children !== 'string') return children;
    
    const lines = children.split('\n');
    
    return lines.map((line, index) => {
      const trimmedLine = line.trim();
      
      // Empty line
      if (!trimmedLine) {
        return <div key={index} style={{ height: '0.8em' }} />;
      }
      
      // Check for Example heading
      if (trimmedLine.toLowerCase().includes('example')) {
        // Process the example line to handle any math expressions
        const exampleParts = [];
        let currentPart = '';
        let inMath = false;
        
        for (let i = 0; i < trimmedLine.length; i++) {
          if (trimmedLine[i] === '\\' && trimmedLine[i+1] === '$') {
            currentPart += '$';
            i++;
            continue;
          }
          
          if (trimmedLine[i] === '$' && (i === 0 || trimmedLine[i-1] !== '\\')) {
            if (currentPart) {
              exampleParts.push({
                type: inMath ? 'math' : 'text',
                content: currentPart.trim()
              });
              currentPart = '';
            }
            inMath = !inMath;
          } else {
            currentPart += trimmedLine[i];
          }
        }
        
        if (currentPart) {
          exampleParts.push({
            type: inMath ? 'math' : 'text',
            content: currentPart.trim()
          });
        }
        
        return (
          <div 
            key={index} 
            style={{
              fontSize: '17px',
              fontWeight: '700',
              color: '#1e40af',
              margin: '1.5em 0 1em 0',
              padding: '0.8em 1em',
              borderBottom: '3px solid #3b82f6',
              background: 'linear-gradient(to right, #eff6ff, transparent)',
              borderRadius: '4px'
            }}
          >
            {exampleParts.map((part, partIndex) => {
              if (part.type === 'math') {
                return (
                  <MathJax 
                    key={partIndex}
                    inline 
                    dynamic
                    style={{
                      display: 'inline',
                      padding: '0 2px',
                      verticalAlign: 'middle'
                    }}
                  >
                    {part.content}
                  </MathJax>
                );
              }
              return <span key={partIndex}>{part.content}</span>;
            })}
          </div>
        );
      }
      
      // Check for Concept heading
      if (trimmedLine.toLowerCase().includes('concept:')) {
        return (
          <div 
            key={index} 
            style={{
              fontSize: '16px',
              fontWeight: '600',
              color: '#0f766e',
              margin: '1.5em 0 0.8em 0',
              paddingBottom: '0.4em',
              borderBottom: '2px solid #14b8a6',
              background: 'linear-gradient(to right, #f0fdfa, transparent)',
              padding: '0.5em 0.8em',
              borderRadius: '4px'
            }}
          >
            {trimmedLine}
          </div>
        );
      }
      
      // Check for numbered step
      const stepMatch = trimmedLine.match(/^(\d+)\.\s*(.*)$/);
      
      if (stepMatch) {
        const [, stepNum, stepContent] = stepMatch;
        const parts = formatInlineMath(stepContent);
        
        return (
          <div 
            key={index} 
            style={{ 
              margin: '1em 0',
              padding: '1em',
              backgroundColor: '#f8fafc',
              borderLeft: '4px solid #3b82f6',
              borderRadius: '0 6px 6px 0',
              boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
            }}
          >
            <div style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: '12px'
            }}>
              <div style={{
                flexShrink: 0,
                width: '32px',
                height: '32px',
                borderRadius: '6px',
                background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '14px',
                fontWeight: '700',
                boxShadow: '0 2px 4px rgba(59,130,246,0.3)'
              }}>
                {stepNum}
              </div>
              <div style={{ 
                flex: 1,
                color: '#1e293b',
                fontSize: '15px',
                lineHeight: '1.8',
                paddingTop: '4px'
              }}>
                {parts.map((part, partIndex) => {
                  if (part.type === 'math') {
                    // Only render MathJax if content exists
                    if (!part.content) return null;
                    return (
                      <MathJax 
                        key={partIndex}
                        inline 
                        dynamic
                        style={{
                          display: 'inline',
                          padding: '0 2px',
                          verticalAlign: 'middle'
                        }}
                      >
                        {part.content.startsWith('$') ? part.content : `$${part.content}$`}
                      </MathJax>
                    );
                  }
                  return (
                    <span key={partIndex} style={{ whiteSpace: 'pre-wrap' }}>
                      {part.content}
                    </span>
                  );
                })}
              </div>
            </div>
          </div>
        );
      }
      
      // Check for bullet point
      const bulletMatch = trimmedLine.match(/^[•\-\*]\s*(.*)$/);
      
      if (bulletMatch) {
        const [, content] = bulletMatch;
        const parts = formatInlineMath(content);
        
        return (
          <div 
            key={index} 
            style={{ 
              margin: '0.8em 0',
              paddingLeft: '1.5em',
              position: 'relative'
            }}
          >
            <span style={{
              position: 'absolute',
              left: '0',
              top: '0.3em',
              color: '#3b82f6',
              fontSize: '18px',
              fontWeight: 'bold'
            }}>•</span>
            <div style={{
              color: '#1e293b',
              fontSize: '15px',
              lineHeight: '1.8'
            }}>
              {parts.map((part, partIndex) => {
                if (part.type === 'math') {
                  return (
                    <MathJax 
                      key={partIndex}
                      inline 
                      dynamic
                      style={{
                        display: 'inline',
                        padding: '0 4px'
                      }}
                    >
                      {`$${part.content}$`}
                    </MathJax>
                  );
                }
                return (
                  <span key={partIndex}>
                    {part.content}
                  </span>
                );
              })}
            </div>
          </div>
        );
      }
      
      // Regular paragraph
      const parts = formatInlineMath(trimmedLine);
      
      return (
        <div 
          key={index} 
          style={{ 
            margin: '0.8em 0',
            color: '#1e293b',
            fontSize: '15px',
            lineHeight: '1.9',
            paddingLeft: '0.5em'
          }}
        >
          {parts.map((part, partIndex) => {
            if (part.type === 'math') {
              return (
                <MathJax 
                  key={partIndex}
                  inline 
                  dynamic
                  style={{
                    display: 'inline',
                    padding: '0 4px'
                  }}
                >
                  {`$${part.content}$`}
                </MathJax>
              );
            }
            return (
              <span key={partIndex}>
                {part.content}
              </span>
            );
          })}
        </div>
      );
    });
  }, [children]);

  return (
    <div style={{ 
      fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      maxWidth: '1400px',
      margin: '0 auto',
      padding: '1.5em',
      backgroundColor: 'white',
      borderRadius: '12px',
      boxShadow: '0 4px 6px rgba(0,0,0,0.07)',
      lineHeight: '1.8'
    }}>
      {processedContent}
    </div>
  );
};

export default MathText;