export function exactMatch(predicted: string, gold: string): boolean {
  if (!gold.trim()) return false;
  
  const normalize = (text: string) => {
    return text
      .toLowerCase()
      .trim()
      .replace(/\b(a|an|the)\b/g, " ")
      .replace(/[^\w\s]/g, "")
      .replace(/\s+/g, " ")
      .trim();
  };

  const predNorm = normalize(predicted);
  const goldNorm = normalize(gold);

  return predNorm.includes(goldNorm);
}
